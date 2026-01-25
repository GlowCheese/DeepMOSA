####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'test_file.txt'
    var_5 = 'test_dir/test_file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'invalid.zip'
    var_17 = var_14 / var_16
    var_18 = 'not a zip file'
    var_19 = module_1.str(var_17)
    var_20 = False
    var_21 = 'test.txt'
    var_22 = var_19 / var_21
    var_23 = 'test_dir/test.txt'
    var_24 = b'test_password'
    var_25 = 'protected.zip'
    var_26 = var_19 / var_25
    var_27 = module_1.str(var_26)
    var_28 = False
    var_29 = 'test_password'
    var_30 = module_0.exists()



# Parsed testcases at query #2
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content in zip'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'protected.zip'
    var_14 = var_8 / var_13
    var_15 = 'test.py'
    var_16 = b'secret'
    var_17 = module_1.str(var_14)
    var_18 = False
    var_19 = 'secret'
    var_20 = module_0.exists()
    var_21 = module_1.str(var_14)
    var_22 = False
    var_23 = 'wrong'
    var_24 = 'invalid.zip'
    var_25 = var_21 / var_24
    var_26 = 'not a zip file'
    var_27 = module_1.str(var_25)
    var_28 = False
    var_29 = 'empty.zip'
    var_30 = var_27 / var_29
    var_31 = module_1.str(var_30)
    var_32 = False
    var_33 = 'no_dir.zip'
    var_34 = var_31 / var_33
    var_35 = 'file.txt'
    var_36 = 'content'
    var_37 = module_1.str(var_34)
    var_38 = False



# Parsed testcases at query #3
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'invalid.zip'
    var_13 = var_8 / var_12
    var_14 = b'not a zip file'
    var_15 = module_1.str(var_13)
    var_16 = False
    var_17 = 'empty.zip'
    var_18 = var_15 / var_17
    var_19 = module_1.str(var_18)
    var_20 = False
    var_21 = 'file.txt'
    var_22 = 'test_repo/file.txt'
    var_23 = b'test_password'
    var_24 = 'protected.zip'
    var_25 = module_1.str(var_20)
    var_26 = False
    var_27 = module_0.exists()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x01\x00\x00\x00test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_11 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_12 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_13 = b'\x00\x00\x00\x00\x00test_dir/\x00PK\x05\x06\x00\x00\x00\x00'
    var_14 = b'\x01\x00\x01\x00\x00\x00\x00\x00'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = 'test_password.zip'
    var_18 = 'test_dir/'
    var_19 = ''
    var_20 = 'test_dir/file.txt'
    var_21 = 'test content'
    var_22 = b'password'
    var_23 = False
    var_24 = 'password'
    var_25 = False
    var_26 = 'wrong_password'
    var_27 = 'test_invalid.zip'
    var_28 = 'not a zip file'
    var_29 = False
    var_30 = 'test_empty.zip'
    var_31 = False
    var_32 = 'test_no_dir.zip'
    var_33 = 'file.txt'
    var_34 = 'test content'
    var_35 = False



# Parsed testcases at query #5
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'test_dir/file.txttest content'
    var_11 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'invalid.zip'
    var_16 = b'not a zip file'
    var_17 = False
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'no_dir.zip'
    var_21 = 'file.txt'
    var_22 = 'test content'
    var_23 = False
    var_24 = 'protected.zip'
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'test content'
    var_29 = b'secret'
    var_30 = False
    var_31 = 'secret'
    var_32 = 'file.txt'
    var_33 = module_0.exists(var_29)
    var_34 = False
    var_35 = 'wrong'
    var_36 = 'wrong'
    var_37 = False



# Parsed testcases at query #6
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_11 = b'test_repo/file.txttest contentPK\x05\x06\x00\x00\x00\x00'
    var_12 = b'\x01\x00\x01\x00\x14\x00\x00\x00\x08\x00\x00\x00'
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = module_0.exists()
    var_16 = 'test_repo'
    var_17 = var_8 / var_16
    var_18 = 'file.txt'
    var_19 = var_17 / var_18
    var_20 = 'test content'
    var_21 = 'test_password.zip'
    var_22 = 'file.txt'
    var_23 = var_17 / var_22
    var_24 = 'test_repo/file.txt'
    var_25 = b'secret'
    var_26 = False
    var_27 = 'secret'
    var_28 = module_0.exists()
    var_29 = False
    var_30 = 'wrong'
    var_31 = 'invalid.zip'
    var_32 = var_22 / var_31
    var_33 = 'not a zip file'
    var_34 = module_1.str(var_32)
    var_35 = False
    var_36 = 'empty.zip'
    var_37 = var_34 / var_36
    var_38 = module_1.str(var_37)
    var_39 = False
    var_40 = 'no_dir.zip'
    var_41 = var_38 / var_40
    var_42 = 'file.txt'
    var_43 = 'content'
    var_44 = module_1.str(var_41)
    var_45 = False



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_8 = b'test content'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = 'empty.zip'
    var_12 = False
    var_13 = 'invalid.zip'
    var_14 = 'file.txt'
    var_15 = 'test content'
    var_16 = False
    var_17 = 'protected.zip'
    var_18 = 'test_dir/'
    var_19 = ''
    var_20 = 'test_dir/file.txt'
    var_21 = 'test content'
    var_22 = b'password'
    var_23 = False
    var_24 = 'password'
    var_25 = False
    var_26 = 'wrong'
    var_27 = 'invalid.zip'
    var_28 = b'not a zip file'
    var_29 = False



# Parsed testcases at query #8
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'invalid.zip'
    var_17 = var_14 / var_16
    var_18 = 'not a zip file'
    var_19 = module_1.str(var_17)
    var_20 = False
    var_21 = 'protected.zip'
    var_22 = var_19 / var_21
    var_23 = 'test.txt'
    var_24 = 'content'
    var_25 = module_1.str(var_22)
    var_26 = False
    var_27 = 'password'
    var_28 = module_0.exists()



# Parsed testcases at query #9
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'protected'
    var_14 = var_8 / var_13
    var_15 = 'secret.txt'
    var_16 = var_14 / var_15
    var_17 = 'secret content'
    var_18 = 'protected.zip'
    var_19 = 'secret.txt'
    var_20 = var_14 / var_19
    var_21 = 'protected/secret.txt'
    var_22 = b'secret'
    var_23 = False
    var_24 = 'secret'
    var_25 = module_0.exists()
    var_26 = False
    var_27 = 'wrong'
    var_28 = 'invalid.zip'
    var_29 = var_19 / var_28
    var_30 = b'not a zip file'
    var_31 = module_1.str(var_29)
    var_32 = False
    var_33 = 'empty.zip'
    var_34 = var_31 / var_33
    var_35 = module_1.str(var_34)
    var_36 = False
    var_37 = 'no_dir.zip'
    var_38 = var_35 / var_37
    var_39 = 'file.txt'
    var_40 = 'content'
    var_41 = module_1.str(var_38)
    var_42 = False



# Parsed testcases at query #10
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'empty.zip'
    var_14 = var_8 / var_13
    var_15 = module_1.str(var_14)
    var_16 = False
    var_17 = 'bad.zip'
    var_18 = var_15 / var_17
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = module_1.str(var_18)
    var_22 = False
    var_23 = 'protected.zip'
    var_24 = var_21 / var_23
    var_25 = 'test_repo/file.txt'
    var_26 = 'secret content'
    var_27 = b'password'
    var_28 = module_1.str(var_24)
    var_29 = False
    var_30 = 'password'
    var_31 = module_0.exists()
    var_32 = module_1.str(var_24)
    var_33 = False
    var_34 = 'wrong'
    var_35 = module_1.str(var_24)
    var_36 = False
    var_37 = True
    var_38 = 'invalid.zip'
    var_39 = var_35 / var_38
    var_40 = b'not a zip file'
    var_41 = module_1.str(var_39)
    var_42 = False



# Parsed testcases at query #11
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = b'PK\x03\x04...'
    var_7 = 'http://example.com/test.zip'
    var_8 = True
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'not_a_zip.txt'
    var_12 = 'This is not a zip file'
    var_13 = False
    var_14 = 'protected_dir/'
    var_15 = ''
    var_16 = 'protected_dir/file.txt'
    var_17 = 'content'
    var_18 = 'correct_password'
    var_19 = 'utf-8'
    var_20 = module_0.encode(var_19)
    var_21 = 'protected.zip'
    var_22 = False
    var_23 = 'protected_dir/'
    var_24 = ''
    var_25 = 'protected_dir/file.txt'
    var_26 = 'content'
    var_27 = 'correct_password'
    var_28 = 'utf-8'
    var_29 = module_0.encode(var_28)
    var_30 = 'protected.zip'
    var_31 = False



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x01\x00\x00\x00test_dir/file.txt\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x01\x00\x00\x00test content\x00\x00\x00'
    var_11 = 'http://example.com/test.zip'
    var_12 = True
    var_13 = 'file.txt'
    var_14 = 'empty.zip'
    var_15 = False
    var_16 = 'invalid.zip'
    var_17 = b'This is not a zip file'
    var_18 = False
    var_19 = 'protected.zip'
    var_20 = 'test_dir/'
    var_21 = ''
    var_22 = 'test_dir/file.txt'
    var_23 = 'test content'
    var_24 = b'secret'
    var_25 = False
    var_26 = 'secret'
    var_27 = False
    var_28 = 'wrong'
    var_29 = False
    var_30 = True



# Parsed testcases at query #13
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'empty.zip'
    var_14 = var_8 / var_13
    var_15 = module_1.str(var_14)
    var_16 = False
    var_17 = 'invalid.zip'
    var_18 = var_15 / var_17
    var_19 = 'not a zip file'
    var_20 = module_1.str(var_18)
    var_21 = False
    var_22 = 'protected.zip'
    var_23 = var_20 / var_22
    var_24 = 'test.txt'
    var_25 = 'test_repo/test.txt'
    var_26 = module_1.str(var_23)
    var_27 = False
    var_28 = 'correct_password'
    var_29 = module_0.exists()



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_11 = b'test_dir/file.txttest content'
    var_12 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = 'file.txt'
    var_16 = 'test_protected.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'test content'
    var_21 = b'secret'
    var_22 = False
    var_23 = True
    var_24 = 'secret'
    var_25 = 'file.txt'
    var_26 = 'test_invalid.zip'
    var_27 = 'This is not a zip file'
    var_28 = False
    var_29 = True
    var_30 = 'test_empty.zip'
    var_31 = False
    var_32 = True
    var_33 = 'test_no_dir.zip'
    var_34 = 'file.txt'
    var_35 = 'test content'
    var_36 = False
    var_37 = True



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False



# Parsed testcases at query #16
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = module_0.exists(var_4)
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00test_dir/file.txtcontent'
    var_11 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'invalid.zip'
    var_16 = b'not a zip file'
    var_17 = False
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'no_dir.zip'
    var_21 = 'file.txt'
    var_22 = 'content'
    var_23 = False
    var_24 = 'protected.zip'
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'content'
    var_29 = b'password'
    var_30 = False
    var_31 = 'password'
    var_32 = False
    var_33 = 'wrong'
    var_34 = False
    var_35 = True



# Parsed testcases at query #17
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_1.str(var_17)
    var_21 = False
    var_22 = 'invalid.zip'
    var_23 = var_20 / var_22
    var_24 = 'not a zip file'
    var_25 = module_1.str(var_23)
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = var_25 / var_27
    var_29 = 'file.txt'
    var_30 = 'content'
    var_31 = b'password'
    var_32 = module_1.str(var_28)
    var_33 = False
    var_34 = 'password'
    var_35 = module_0.exists()
    var_36 = module_1.str(var_28)
    var_37 = False
    var_38 = 'wrong'
    var_39 = module_1.str(var_28)
    var_40 = False
    var_41 = True



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04...'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'invalid.zip'
    var_11 = b'not a zip file'
    var_12 = False
    var_13 = 'empty.zip'
    var_14 = False
    var_15 = 'no_dir.zip'
    var_16 = 'file.txt'
    var_17 = 'content'
    var_18 = False
    var_19 = 'protected.zip'
    var_20 = 'test_dir/'
    var_21 = ''
    var_22 = 'test_dir/file.txt'
    var_23 = 'content'
    var_24 = b'password'
    var_25 = False
    var_26 = 'password'
    var_27 = False
    var_28 = 'wrong'
    var_29 = False
    var_30 = True



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    assert var_1 == 'test content'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = b'PK\x03\x04'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'invalid.zip'
    var_13 = 'not a zip file'
    var_14 = False
    var_15 = True
    var_16 = 'empty.zip'
    var_17 = False
    var_18 = True
    var_19 = 'no_dir.zip'
    var_20 = 'file.txt'
    var_21 = 'test content'
    var_22 = False
    var_23 = True
    var_24 = 'password.zip'
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'test content'
    var_29 = b'secret'
    var_30 = False
    var_31 = True
    var_32 = 'secret'
    var_33 = False
    var_34 = True
    var_35 = 'wrong'
    var_36 = False
    var_37 = True



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'



# Parsed testcases at query #21
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = module_0.exists(var_4)
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = 'invalid.zip'
    var_12 = 'not a zip file'
    var_13 = False
    var_14 = 'empty.zip'
    var_15 = False
    var_16 = 'protected.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'content'
    var_21 = b'password'
    var_22 = False
    var_23 = 'password'
    var_24 = False
    var_25 = 'wrong'
    var_26 = False
    var_27 = True



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/valid.zip'
    var_1 = True
    var_2 = 'test.zip'
    var_3 = 'test/'
    var_4 = ''
    var_5 = 'test/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'file.txt'
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = True
    var_13 = 'no_dir.zip'
    var_14 = 'file.txt'
    var_15 = 'content'
    var_16 = False
    var_17 = True
    var_18 = 'protected.zip'
    var_19 = 'test/'
    var_20 = ''
    var_21 = 'test/file.txt'
    var_22 = 'content'
    var_23 = b'secret'
    var_24 = False
    var_25 = True
    var_26 = 'secret'
    var_27 = 'protected.zip'
    var_28 = 'test/'
    var_29 = ''
    var_30 = 'test/file.txt'
    var_31 = 'content'
    var_32 = b'secret'
    var_33 = False
    var_34 = True
    var_35 = 'wrong'
    var_36 = 'corrupted.zip'
    var_37 = b'not a zip file'
    var_38 = False
    var_39 = True



# Parsed testcases at query #23
#--------------------------


import requests.api as module_0
import genericpath as module_1
import locale as module_2
import email._encoded_words as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = 'http://example.com/test.zip'
    var_6 = module_0.get(var_5)
    var_7 = True
    var_8 = module_1.exists()
    var_9 = 'file.txt'
    var_10 = module_1.exists()
    var_11 = 'local_test.zip'
    var_12 = var_5 / var_11
    var_13 = 'local_test_dir/'
    var_14 = ''
    var_15 = 'local_test_dir/local_file.txt'
    var_16 = 'local content'
    var_17 = module_2.str(var_12)
    var_18 = False
    var_19 = True
    var_20 = module_1.exists()
    var_21 = 'local_file.txt'
    var_22 = var_9 / var_21
    var_23 = module_1.exists()
    var_24 = 'protected.zip'
    var_25 = var_13 / var_24
    var_26 = 'protected_dir/'
    var_27 = ''
    var_28 = 'protected_dir/protected_file.txt'
    var_29 = 'protected content'
    var_30 = 'secret'
    var_31 = 'utf-8'
    var_32 = module_3.encode(var_31)
    var_33 = module_2.str(var_25)
    var_34 = False
    var_35 = True
    var_36 = 'secret'
    var_37 = module_1.exists()
    var_38 = var_32.name
    assert var_38 == 'protected_dir'
    var_39 = 'protected_file.txt'
    var_40 = var_21 / var_39
    var_41 = module_1.exists()
    var_42 = 'invalid.zip'
    var_43 = var_26 / var_42
    var_44 = 'not a zip file'
    var_45 = module_2.str(var_43)
    var_46 = False
    var_47 = True
    var_48 = 'empty.zip'
    var_49 = var_45 / var_48
    var_50 = module_2.str(var_49)
    var_51 = False
    var_52 = True
    var_53 = 'no_dir.zip'
    var_54 = var_50 / var_53
    var_55 = 'file.txt'
    var_56 = 'content'
    var_57 = module_2.str(var_54)
    var_58 = False
    var_59 = True



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00test_dir/'
    var_10 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_11 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_12 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00\x18\x00\x00\x00'
    var_13 = b'\x00\x00\x00\x00\x00\x00'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = 'invalid.zip'
    var_17 = 'not a zip file'
    var_18 = False
    var_19 = 'empty.zip'
    var_20 = False
    var_21 = 'no_dir.zip'
    var_22 = 'file.txt'
    var_23 = 'test content'
    var_24 = False
    var_25 = 'password.zip'
    var_26 = 'test_dir/'
    var_27 = ''
    var_28 = 'test_dir/file.txt'
    var_29 = 'test content'
    var_30 = b'secret'
    var_31 = False
    var_32 = 'secret'
    var_33 = False
    var_34 = 'wrong'
    var_35 = False
    var_36 = True



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = b'PK\x03\x04...'
    var_9 = None
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'protected.zip'
    var_13 = 'test_dir/'
    var_14 = ''
    var_15 = 'test_dir/file.txt'
    var_16 = 'test content'
    var_17 = b'secret'
    var_18 = False
    var_19 = True
    var_20 = 'secret'
    var_21 = 'invalid.zip'
    var_22 = 'not a zip file'
    var_23 = False
    var_24 = True
    var_25 = 'empty.zip'
    var_26 = False
    var_27 = True
    var_28 = 'no_dir.zip'
    var_29 = 'file.txt'
    var_30 = 'test content'
    var_31 = False
    var_32 = True



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'invalid.zip'
    var_13 = 'not a zip file'
    var_14 = False
    var_15 = True
    var_16 = 'empty.zip'
    var_17 = False
    var_18 = True
    var_19 = 'no_dir.zip'
    var_20 = 'file.txt'
    var_21 = 'test content'
    var_22 = False
    var_23 = True



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'invalid.zip'
    var_11 = 'not a zip file'
    var_12 = False
    var_13 = True
    var_14 = 'empty.zip'
    var_15 = False
    var_16 = True
    var_17 = 'no_dir.zip'
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = False
    var_21 = True



# Parsed testcases at query #3
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'test data'
    var_1 = 'test_dir/'
    var_2 = 'http://example.com/test.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3)
    assert var_4 == '/tmp/test/test_dir'
    var_5 = 'test_dir/'
    var_6 = '/path/to/test.zip'
    var_7 = False
    var_8 = module_0.unzip(var_6, var_7)
    assert var_8 == '/tmp/test/test_dir'
    var_9 = b'test data'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.unzip(var_10, var_11)
    var_13 = b'test data'
    var_14 = 'test_file.txt'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = module_0.unzip(var_15, var_16)
    var_18 = b'test data'
    var_19 = 'test_dir/'
    var_20 = None
    var_21 = 'http://example.com/test.zip'
    var_22 = True
    var_23 = 'correct_password'
    var_24 = module_0.unzip(var_21, var_22, password=var_23)
    assert var_24 == '/tmp/test/test_dir'
    var_25 = b'test data'
    var_26 = 'test_dir/'
    var_27 = 'http://example.com/test.zip'
    var_28 = True
    var_29 = 'incorrect_password'
    var_30 = module_0.unzip(var_27, var_28, password=var_29)
    var_31 = b'test data'
    var_32 = 'http://example.com/test.zip'
    var_33 = True
    var_34 = module_0.unzip(var_32, var_33)



# Parsed testcases at query #4
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'file.txt'
    var_8 = module_0.exists()
    var_9 = b'PK\x03\x04...'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'empty.zip'
    var_14 = var_9 / var_13
    var_15 = module_1.str(var_14)
    var_16 = False
    var_17 = 'no_dir.zip'
    var_18 = var_15 / var_17
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = module_1.str(var_18)
    var_22 = False
    var_23 = 'protected.zip'
    var_24 = var_21 / var_23
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'content'
    var_29 = b'secret'
    var_30 = module_1.str(var_24)
    var_31 = False
    var_32 = 'secret'
    var_33 = module_0.exists()
    var_34 = module_1.str(var_24)
    var_35 = False
    var_36 = 'wrong'
    var_37 = module_1.str(var_24)
    var_38 = False
    var_39 = True
    var_40 = 'invalid.zip'
    var_41 = var_37 / var_40
    var_42 = 'not a zip file'
    var_43 = module_1.str(var_41)
    var_44 = False



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/valid-repo.zip'
    var_1 = True
    var_2 = 'test.zip'
    var_3 = 'test/'
    var_4 = ''
    var_5 = False
    var_6 = True
    var_7 = 'empty.zip'
    var_8 = False
    var_9 = True
    var_10 = 'no-dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = True
    var_15 = 'protected.zip'
    var_16 = 'test/'
    var_17 = ''
    var_18 = b'secret'
    var_19 = False
    var_20 = True
    var_21 = 'secret'
    var_22 = False
    var_23 = True
    var_24 = 'wrong'
    var_25 = 'invalid.zip'
    var_26 = 'not a zip file'
    var_27 = False
    var_28 = True



# Parsed testcases at query #6
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04...'
    var_9 = None
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'protected.zip'
    var_14 = var_8 / var_13
    var_15 = 'test.txt'
    var_16 = var_8 / var_15
    var_17 = 'test_repo/test.txt'
    var_18 = b'secret'
    var_19 = module_1.str(var_14)
    var_20 = False
    var_21 = 'secret'
    var_22 = module_0.exists()
    var_23 = module_1.str(var_14)
    var_24 = False
    var_25 = 'wrong'
    var_26 = 'invalid.zip'
    var_27 = var_23 / var_26
    var_28 = 'not a zip file'
    var_29 = module_1.str(var_27)
    var_30 = False
    var_31 = 'empty.zip'
    var_32 = var_29 / var_31
    var_33 = module_1.str(var_32)
    var_34 = False
    var_35 = 'bad.zip'
    var_36 = var_33 / var_35
    var_37 = 'file.txt'
    var_38 = 'content'
    var_39 = module_1.str(var_36)
    var_40 = False



# Parsed testcases at query #7
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'file.txt'
    var_8 = b'PK\x03\x04...'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'empty.zip'
    var_13 = var_9 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_1.str(var_17)
    var_21 = False
    var_22 = 'protected.zip'
    var_23 = var_20 / var_22
    var_24 = 'test_dir/'
    var_25 = ''
    var_26 = 'test_dir/file.txt'
    var_27 = 'content'
    var_28 = b'secret'
    var_29 = module_1.str(var_23)
    var_30 = False
    var_31 = 'secret'
    var_32 = module_0.exists()
    var_33 = module_1.str(var_23)
    var_34 = False
    var_35 = 'wrong'
    var_36 = module_1.str(var_23)
    var_37 = False
    var_38 = True
    var_39 = 'invalid.zip'
    var_40 = var_36 / var_39
    var_41 = 'not a zip file'
    var_42 = module_1.str(var_40)
    var_43 = False



# Parsed testcases at query #8
#--------------------------


import cookiecutter.zipfile as module_0
import genericpath as module_1
import email._encoded_words as module_2

def test_case_0():
    var_0 = 'https://example.com/nonexistent.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = 'test.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = module_0.unzip(var_3, var_5)
    var_7 = False
    var_8 = module_0.unzip(var_3, var_7)
    var_9 = 'test.txt'
    var_10 = 'test content'
    var_11 = False
    var_12 = module_0.unzip(var_9, var_11)
    var_13 = 'test_dir/'
    var_14 = ''
    var_15 = 'test_dir/test.txt'
    var_16 = 'test content'
    var_17 = False
    var_18 = module_0.unzip(var_13, var_17)
    var_19 = module_1.exists(var_18)
    var_20 = 'test.txt'
    var_21 = module_1.exists(var_16)
    var_22 = 'test_dir/'
    var_23 = ''
    var_24 = 'test_dir/test.txt'
    var_25 = 'test content'
    var_26 = 'password'
    var_27 = 'utf-8'
    var_28 = module_2.encode(var_27)
    var_29 = False
    var_30 = 'wrong_password'
    var_31 = module_0.unzip(var_22, var_29, password=var_30)
    var_32 = False
    var_33 = 'password'
    var_34 = module_0.unzip(var_22, var_32, password=var_33)
    var_35 = module_1.exists(var_34)
    var_36 = 'test.txt'
    var_37 = module_1.exists(var_21)
    var_38 = 'test_dir/'
    var_39 = ''
    var_40 = 'test_dir/test.txt'
    var_41 = 'test content'
    var_42 = 'password'
    var_43 = 'utf-8'
    var_44 = module_2.encode(var_43)
    var_45 = False
    var_46 = True
    var_47 = module_0.unzip(var_38, var_45, no_input=var_46)
    var_48 = b'not a zip file'
    var_49 = False
    var_50 = module_0.unzip(var_48, var_49)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04'
    var_8 = b'test content'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = 'invalid.zip'
    var_12 = 'not a zip file'
    var_13 = False
    var_14 = 'empty.zip'
    var_15 = False
    var_16 = 'no_dir.zip'
    var_17 = 'file.txt'
    var_18 = 'test content'
    var_19 = False
    var_20 = 'protected.zip'
    var_21 = 'test_dir/'
    var_22 = ''
    var_23 = 'test_dir/file.txt'
    var_24 = 'test content'
    var_25 = b'test_pass'
    var_26 = False
    var_27 = 'test_pass'
    var_28 = 'protected.zip'
    var_29 = 'test_dir/'
    var_30 = ''
    var_31 = 'test_dir/file.txt'
    var_32 = 'test content'
    var_33 = b'test_pass'
    var_34 = False
    var_35 = 'wrong_pass'



# Parsed testcases at query #10
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.exists()
    var_13 = 'protected'
    var_14 = var_8 / var_13
    var_15 = 'secret.txt'
    var_16 = var_14 / var_15
    var_17 = 'secret content'
    var_18 = 'protected.zip'
    var_19 = 'secret.txt'
    var_20 = var_14 / var_19
    var_21 = 'protected/secret.txt'
    var_22 = b'secret'
    var_23 = False
    var_24 = 'secret'
    var_25 = module_0.exists()
    var_26 = False
    var_27 = 'wrong'
    var_28 = 'invalid.zip'
    var_29 = var_19 / var_28
    var_30 = 'not a zip file'
    var_31 = module_1.str(var_29)
    var_32 = False
    var_33 = 'empty.zip'
    var_34 = var_31 / var_33
    var_35 = module_1.str(var_34)
    var_36 = False
    var_37 = 'no_dir.zip'
    var_38 = var_35 / var_37
    var_39 = 'file.txt'
    var_40 = 'content'
    var_41 = module_1.str(var_38)
    var_42 = False



# Parsed testcases at query #11
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'https://example.com/valid.zip'
    var_1 = True
    var_2 = 'test.zip'
    var_3 = 'test_dir/'
    var_4 = ''
    var_5 = 'test_dir/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'file.txt'
    var_9 = module_0.exists(var_6)
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'no_dir.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
    var_15 = False
    var_16 = 'protected.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'content'
    var_21 = False
    var_22 = True
    var_23 = False
    var_24 = 'correct_password'
    var_25 = 'invalid.zip'
    var_26 = 'not a zip file'
    var_27 = False



# Parsed testcases at query #12
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = True
    var_7 = 'file.txt'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'file.txt'
    var_11 = module_0.exists(var_7)
    var_12 = 'test_password.zip'
    var_13 = 'test_dir/'
    var_14 = ''
    var_15 = 'test_dir/file.txt'
    var_16 = 'test content'
    var_17 = b'test_password'
    var_18 = False
    var_19 = True
    var_20 = 'test_password'
    var_21 = 'file.txt'
    var_22 = False
    var_23 = True
    var_24 = 'wrong_password'
    var_25 = 'test_invalid.zip'
    var_26 = b'This is not a zip file'
    var_27 = False
    var_28 = True
    var_29 = 'test_empty.zip'
    var_30 = False
    var_31 = True
    var_32 = 'test_no_dir.zip'
    var_33 = 'file.txt'
    var_34 = 'test content'
    var_35 = False
    var_36 = True



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00'
    var_9 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00'
    var_10 = b'\x00\x00\x00!\x00\x00\x00\x00\x00'
    var_11 = b'\x00\x00\x00\x00\x00\x00\x00test_dir/'
    var_12 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00'
    var_13 = b'\x01\x00\x12\x00\x00\x00\x0c\x00'
    var_14 = b'\x00\x00\x00\x00'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = 'empty.zip'
    var_18 = False
    var_19 = 'invalid.zip'
    var_20 = 'not a zip file'
    var_21 = False
    var_22 = 'password.zip'
    var_23 = 'test_dir/'
    var_24 = ''
    var_25 = 'test_dir/file.txt'
    var_26 = 'test content'
    var_27 = b'secret'
    var_28 = False
    var_29 = 'secret'
    var_30 = False
    var_31 = 'wrong'
    var_32 = False
    var_33 = True



# Parsed testcases at query #14
#--------------------------


import locale as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'empty.zip'
    var_7 = False
    var_8 = module_0.str(var_5)
    var_9 = 'no_dir.zip'
    var_10 = 'file.txt'
    var_11 = 'content'
    var_12 = False
    var_13 = module_0.str(var_11)
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = module_0.str(var_11)
    var_18 = 'protected.zip'
    var_19 = 'test_dir/'
    var_20 = ''
    var_21 = 'test_dir/file.txt'
    var_22 = 'content'
    var_23 = b'secret'
    var_24 = False
    var_25 = True
    var_26 = module_0.str(var_25)
    var_27 = 'protected.zip'
    var_28 = 'test_dir/'
    var_29 = ''
    var_30 = 'test_dir/file.txt'
    var_31 = 'content'
    var_32 = b'secret'
    var_33 = False
    var_34 = 'secret'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'empty.zip'
    var_7 = False
    var_8 = 'no_dir.zip'
    var_9 = 'file.txt'
    var_10 = 'test content'
    var_11 = False
    var_12 = 'nonexistent.zip'
    var_13 = False
    var_14 = module_0.unzip(var_12, var_13)
    var_15 = 'protected.zip'
    var_16 = 'test_dir/'
    var_17 = ''
    var_18 = 'test_dir/file.txt'
    var_19 = 'test content'
    var_20 = b'password'
    var_21 = False
    var_22 = True
    var_23 = 'protected.zip'
    var_24 = 'test_dir/'
    var_25 = ''
    var_26 = 'test_dir/file.txt'
    var_27 = 'test content'
    var_28 = b'password'
    var_29 = False
    var_30 = 'password'
    var_31 = 'protected.zip'
    var_32 = 'test_dir/'
    var_33 = ''
    var_34 = 'test_dir/file.txt'
    var_35 = 'test content'
    var_36 = b'password'
    var_37 = False
    var_38 = 'wrong_password'
    var_39 = 'invalid.zip'
    var_40 = 'This is not a zip file'
    var_41 = False



