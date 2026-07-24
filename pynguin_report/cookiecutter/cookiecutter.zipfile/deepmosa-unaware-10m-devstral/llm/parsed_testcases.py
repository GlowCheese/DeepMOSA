####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_16 = 'not a zip file'
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
    var_32 = 'file.txt'
    var_33 = module_0.exists(var_13)
    var_34 = False
    var_35 = 'wrong'
    var_36 = False
    var_37 = True



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
    var_24 = var_19 / var_23
    var_25 = 'test_repo/test.txt'
    var_26 = b'secret'
    var_27 = module_1.str(var_22)
    var_28 = False
    var_29 = 'secret'
    var_30 = module_0.exists()
    var_31 = module_1.str(var_22)
    var_32 = False
    var_33 = 'wrong'
    var_34 = module_1.str(var_22)
    var_35 = False
    var_36 = True



# Parsed testcases at query #3
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
    var_6 = True
    var_7 = module_0.exists()
    var_8 = 'file.txt'
    var_9 = module_0.exists()
    var_10 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_11 = b'content'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = module_0.exists()
    var_15 = 'empty.zip'
    var_16 = var_10 / var_15
    var_17 = module_1.str(var_16)
    var_18 = False
    var_19 = True
    var_20 = 'invalid.zip'
    var_21 = var_17 / var_20
    var_22 = 'not a zip file'
    var_23 = module_1.str(var_21)
    var_24 = False
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = var_23 / var_26
    var_28 = 'test_dir/'
    var_29 = ''
    var_30 = 'test_dir/file.txt'
    var_31 = 'content'
    var_32 = b'password'
    var_33 = module_1.str(var_27)
    var_34 = False
    var_35 = True
    var_36 = 'password'
    var_37 = module_0.exists()
    var_38 = module_1.str(var_27)
    var_39 = False
    var_40 = True
    var_41 = 'wrong'
    var_42 = module_1.str(var_27)
    var_43 = False
    var_44 = True



# Parsed testcases at query #4
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
    var_21 = 'protected.zip'
    var_22 = var_19 / var_21
    var_23 = 'test.txt'
    var_24 = var_19 / var_23
    var_25 = module_1.str(var_22)
    var_26 = False
    var_27 = 'wrong'
    var_28 = module_1.str(var_22)
    var_29 = False
    var_30 = 'correct'
    var_31 = module_0.exists()



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
    var_9 = b'\x00\x00\x00\x00test_dir/file.txt'
    var_10 = b'test contentPK\x01\x02\x14\x00\x14\x00'
    var_11 = b'\x00\x00\x00\x00\x00'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'empty.zip'
    var_16 = False
    var_17 = 'invalid.zip'
    var_18 = 'not a zip file'
    var_19 = False
    var_20 = 'password.zip'
    var_21 = 'test_dir/'
    var_22 = ''
    var_23 = 'test_dir/file.txt'
    var_24 = 'test content'
    var_25 = b'password'
    var_26 = False
    var_27 = 'password'
    var_28 = 'file.txt'
    var_29 = module_0.exists(var_25)
    var_30 = False
    var_31 = 'wrong'
    var_32 = False
    var_33 = True



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = True
    var_7 = None
    var_8 = 'file.txt'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = 'http://example.com/test.zip'
    var_8 = True
    var_9 = 'invalid.zip'
    var_10 = 'not a zip file'
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = False
    var_14 = 'no_dir.zip'
    var_15 = 'file.txt'
    var_16 = 'content'
    var_17 = False
    var_18 = 'password.zip'
    var_19 = 'test_dir/'
    var_20 = ''
    var_21 = 'test_dir/file.txt'
    var_22 = 'content'
    var_23 = b'password'
    var_24 = False
    var_25 = 'password'
    var_26 = 'file.txt'
    var_27 = False
    var_28 = 'wrong'
    var_29 = False
    var_30 = True



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
    var_12 = 'invalid.zip'
    var_13 = var_8 / var_12
    var_14 = b'not a zip file'
    var_15 = module_1.str(var_13)
    var_16 = False
    var_17 = 'empty.zip'
    var_18 = var_15 / var_17
    var_19 = module_1.str(var_18)
    var_20 = False
    var_21 = 'bad.zip'
    var_22 = var_19 / var_21
    var_23 = 'file.txt'
    var_24 = 'content'
    var_25 = module_1.str(var_22)
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = var_25 / var_27
    var_29 = 'test_repo/file.txt'
    var_30 = 'content'
    var_31 = b'testpass'
    var_32 = module_1.str(var_28)
    var_33 = False
    var_34 = module_0.exists()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'invalid.zip'
    var_7 = 'not a zip file'
    var_8 = False
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'no_dir.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'protected.zip'
    var_16 = 'test_dir/'
    var_17 = ''
    var_18 = 'test_dir/file.txt'
    var_19 = 'content'
    var_20 = b'password'
    var_21 = False
    var_22 = True
    var_23 = 'protected.zip'
    var_24 = 'test_dir/'
    var_25 = ''
    var_26 = 'test_dir/file.txt'
    var_27 = 'content'
    var_28 = b'password'
    var_29 = False
    var_30 = 'password'



# Parsed testcases at query #11
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'requests.get'
    var_1 = module_0.patch(var_0)
    var_2 = b'test'
    var_3 = 'zipfile.ZipFile'
    var_4 = module_0.patch(var_3)
    var_5 = 'test/'
    var_6 = 'test/file'
    var_7 = 'tempfile.mkdtemp'
    var_8 = module_0.patch(var_7)
    var_9 = 'http://test.com/test.zip'
    var_10 = True
    var_11 = False
    var_12 = None
    var_13 = module_0.patch(var_3)
    var_14 = module_0.patch(var_7)
    var_15 = '/path/to/test.zip'
    var_16 = module_0.patch(var_0)
    var_17 = module_0.patch(var_3)
    var_18 = 'http://test.com/test.zip'
    var_19 = True
    var_20 = False
    var_21 = None
    var_22 = module_0.patch(var_18)
    var_23 = module_0.patch(var_20)
    var_24 = 'http://test.com/test.zip'
    var_25 = True
    var_26 = False
    var_27 = None
    var_28 = module_0.patch(var_24)
    var_29 = module_0.patch(var_26)
    var_30 = module_0.patch(var_7)
    var_31 = 'password'
    var_32 = module_0.patch(var_24)
    var_33 = module_0.patch(var_26)
    var_34 = module_0.patch(var_7)
    var_35 = 'http://test.com/test.zip'
    var_36 = True
    var_37 = False
    var_38 = 'wrongpassword'
    var_39 = module_0.patch(var_35)
    var_40 = module_0.patch(var_37)
    var_41 = module_0.patch(var_7)
    var_42 = 'http://test.com/test.zip'
    var_43 = True
    var_44 = None



# Parsed testcases at query #12
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
    var_10 = b'\x00\x00\x00\x00\x01\x00\x00\x00test_dir/file.txtcontent'
    var_11 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'no_dir.zip'
    var_21 = 'file.txt'
    var_22 = 'content'
    var_23 = False
    var_24 = 'password.zip'
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'content'
    var_29 = b'password'
    var_30 = False
    var_31 = 'password'
    var_32 = 'file.txt'
    var_33 = module_0.exists(var_29)
    var_34 = False
    var_35 = 'wrong'
    var_36 = False
    var_37 = True



# Parsed testcases at query #13
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
    var_7 = module_0.exists(var_4)
    var_8 = b'PK\x03\x04'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'empty.zip'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
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
    var_27 = False
    var_28 = True



# Parsed testcases at query #14
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
    var_14 = b'not a zip file'
    var_15 = False
    var_16 = 'protected.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'test content'
    var_21 = b'secret'
    var_22 = False
    var_23 = 'secret'
    var_24 = False
    var_25 = 'wrong'
    var_26 = False
    var_27 = True



# Parsed testcases at query #15
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
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'invalid.zip'
    var_13 = b'not a zip file'
    var_14 = False
    var_15 = 'protected.zip'
    var_16 = 'test_dir/'
    var_17 = ''
    var_18 = 'test_dir/file.txt'
    var_19 = 'content'
    var_20 = b'password'
    var_21 = False
    var_22 = 'password'
    var_23 = False
    var_24 = 'wrong'
    var_25 = False
    var_26 = True
    var_27 = 'no_dir.zip'
    var_28 = 'file.txt'
    var_29 = 'content'
    var_30 = False



# Parsed testcases at query #16
#--------------------------


import cookiecutter.zipfile as module_0
import locale as module_1

def test_case_0():
    var_0 = b'fake content'
    var_1 = 'test_dir/'
    var_2 = 'https://example.com/test.zip'
    var_3 = True
    var_4 = '/tmp'
    var_5 = None
    var_6 = module_0.unzip(var_2, var_3, var_4, var_3, var_5)
    assert var_6 == '/tmp/test/test_dir'
    var_7 = 100
    var_8 = '/tmp/test'
    var_9 = 'test_dir/'
    var_10 = '/path/to/local.zip'
    var_11 = False
    var_12 = '/tmp'
    var_13 = True
    var_14 = None
    var_15 = module_0.unzip(var_10, var_11, var_12, var_13, var_14)
    assert var_15 == '/tmp/test/test_dir'
    var_16 = '/tmp/test'
    var_17 = b'fake content'
    var_18 = 'https://example.com/test.zip'
    var_19 = True
    var_20 = '/tmp'
    var_21 = None
    var_22 = module_0.unzip(var_18, var_19, var_20, var_19, var_21)
    var_23 = module_1.str(var_18)
    var_24 = b'fake content'
    var_25 = 'file.txt'
    var_26 = 'https://example.com/test.zip'
    var_27 = True
    var_28 = '/tmp'
    var_29 = None
    var_30 = module_0.unzip(var_26, var_27, var_28, var_27, var_29)
    var_31 = module_1.str(var_27)
    var_32 = b'fake content'
    var_33 = 'test_dir/'
    var_34 = 'Password required'
    var_35 = 'https://example.com/test.zip'
    var_36 = True
    var_37 = '/tmp'
    var_38 = False
    var_39 = None
    var_40 = module_0.unzip(var_35, var_36, var_37, var_38, var_39)
    assert var_40 == '/tmp/test/test_dir'
    var_41 = 'Repo password'
    var_42 = '/tmp/test'
    var_43 = b'correct_password'
    var_44 = b'fake content'
    var_45 = 'test_dir/'
    var_46 = 'Password required'
    var_47 = 'https://example.com/test.zip'
    var_48 = True
    var_49 = '/tmp'
    var_50 = 'wrong_password'
    var_51 = module_0.unzip(var_47, var_48, var_49, var_48, var_50)
    var_52 = module_1.str(var_49)
    var_53 = b'fake content'
    var_54 = 'Not a zip file'
    var_55 = 'https://example.com/test.zip'
    var_56 = True
    var_57 = '/tmp'
    var_58 = None
    var_59 = module_0.unzip(var_55, var_56, var_57, var_56, var_58)
    var_60 = module_1.str(var_56)



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
    var_22 = b'password'
    var_23 = False
    var_24 = 'password'
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



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------


import zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'invalid.zip'
    var_11 = 'not a zip file'
    var_12 = False
    var_13 = 'empty.zip'
    var_14 = False
    var_15 = 'no_dir.zip'
    var_16 = 'file.txt'
    var_17 = 'content'
    var_18 = False
    var_19 = 'test_dir/'
    var_20 = ''
    var_21 = 'test_dir/file.txt'
    var_22 = 'content'
    var_23 = 'a'
    var_24 = module_0.ZipFile(var_19, var_23)
    var_25 = b'test_password'
    var_26 = var_24.setpassword(var_25)
    var_27 = var_24.close()
    var_28 = False



# Parsed testcases at query #20
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = 'rb'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = module_0.exists()
    var_9 = 'file.txt'
    var_10 = module_0.exists()
    var_11 = 'test.zip'
    var_12 = var_5 / var_11
    var_13 = 'test_dir/'
    var_14 = ''
    var_15 = 'test_dir/file.txt'
    var_16 = 'content'
    var_17 = module_1.str(var_12)
    var_18 = False
    var_19 = True
    var_20 = module_0.exists()
    var_21 = 'file.txt'
    var_22 = module_0.exists()
    var_23 = 'invalid.zip'
    var_24 = var_13 / var_23
    var_25 = 'not a zip file'
    var_26 = module_1.str(var_24)
    var_27 = False
    var_28 = True
    var_29 = 'empty.zip'
    var_30 = var_26 / var_29
    var_31 = module_1.str(var_30)
    var_32 = False
    var_33 = True
    var_34 = 'no_dir.zip'
    var_35 = var_31 / var_34
    var_36 = 'file.txt'
    var_37 = 'content'
    var_38 = module_1.str(var_35)
    var_39 = False
    var_40 = True
    var_41 = 'password.zip'
    var_42 = var_38 / var_41
    var_43 = 'test_dir/'
    var_44 = ''
    var_45 = 'test_dir/file.txt'
    var_46 = 'content'
    var_47 = b'secret'
    var_48 = module_1.str(var_42)
    var_49 = False
    var_50 = True
    var_51 = 'secret'
    var_52 = module_0.exists()
    var_53 = 'file.txt'
    var_54 = var_21 / var_53
    var_55 = module_0.exists()
    var_56 = module_1.str(var_42)
    var_57 = False
    var_58 = True
    var_59 = 'wrong'
    var_60 = module_1.str(var_42)
    var_61 = False
    var_62 = True
    var_63 = None



# Parsed testcases at query #21
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'test content'
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
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.unzip(var_10, var_11)
    var_13 = b'test content'
    var_14 = 'test_file.txt'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = module_0.unzip(var_15, var_16)
    var_18 = b'test content'
    var_19 = 'test_dir/'
    var_20 = 'Password required'
    var_21 = 'http://example.com/test.zip'
    var_22 = True
    var_23 = module_0.unzip(var_21, var_22)
    assert var_23 == '/tmp/test/test_dir'
    var_24 = b'test content'
    var_25 = 'test_dir/'
    var_26 = 'Password required'
    var_27 = 'http://example.com/test.zip'
    var_28 = True
    var_29 = module_0.unzip(var_27, var_28)
    var_30 = b'test content'
    var_31 = 'Invalid zip file'
    var_32 = 'http://example.com/test.zip'
    var_33 = True
    var_34 = module_0.unzip(var_32, var_33)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    assert var_4 == 'test_dir'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'PK\x03\x04...'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'bad.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
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
    var_26 = 'invalid.zip'
    var_27 = 'not a zip file'
    var_28 = False



# Parsed testcases at query #24
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
    var_24 = 'file.txt'
    var_25 = 'test_repo/file.txt'
    var_26 = b'secret'
    var_27 = module_1.str(var_23)
    var_28 = False
    var_29 = 'secret'
    var_30 = module_0.exists()
    var_31 = module_1.str(var_23)
    var_32 = False
    var_33 = 'wrong'
    var_34 = module_1.str(var_23)
    var_35 = False
    var_36 = True



# Parsed testcases at query #25
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
    var_6 = '/path/to/local.zip'
    var_7 = False
    var_8 = module_0.unzip(var_6, var_7)
    assert var_8 == '/tmp/test/test_dir'
    var_9 = b'test data'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = module_0.unzip(var_10, var_11)
    var_13 = b'test data'
    var_14 = 'file.txt'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = module_0.unzip(var_15, var_16)
    var_18 = b'test data'
    var_19 = 'test_dir/'
    var_20 = 'Password required'
    var_21 = 'http://example.com/test.zip'
    var_22 = True
    var_23 = 'correct_password'
    var_24 = module_0.unzip(var_21, var_22, password=var_23)
    assert var_24 == '/tmp/test/test_dir'
    var_25 = b'test data'
    var_26 = 'test_dir/'
    var_27 = 'Password required'
    var_28 = 'http://example.com/test.zip'
    var_29 = True
    var_30 = 'wrong_password'
    var_31 = module_0.unzip(var_28, var_29, password=var_30)
    var_32 = b'test data'
    var_33 = 'Invalid zip file'
    var_34 = 'http://example.com/test.zip'
    var_35 = True
    var_36 = module_0.unzip(var_34, var_35)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_6 = True
    var_7 = module_0.exists()
    var_8 = 'file.txt'
    var_9 = module_0.exists()
    var_10 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_11 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_12 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00test_dir/'
    var_13 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = module_0.exists()
    var_17 = 'protected.zip'
    var_18 = var_10 / var_17
    var_19 = 'test_dir/'
    var_20 = ''
    var_21 = 'test_dir/file.txt'
    var_22 = 'content'
    var_23 = b'password'
    var_24 = module_1.str(var_18)
    var_25 = False
    var_26 = True
    var_27 = 'password'
    var_28 = module_0.exists()
    var_29 = 'empty.zip'
    var_30 = var_19 / var_29
    var_31 = module_1.str(var_30)
    var_32 = False
    var_33 = True
    var_34 = 'no_dir.zip'
    var_35 = var_33 / var_34
    var_36 = 'file.txt'
    var_37 = 'content'
    var_38 = module_1.str(var_35)
    var_39 = False
    var_40 = True
    var_41 = 'invalid.zip'
    var_42 = var_38 / var_41
    var_43 = module_1.str(var_42)
    var_44 = False
    var_45 = True



# Parsed testcases at query #2
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
    var_9 = b'\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_11 = b'test_dir/'
    var_12 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_13 = b'test_dir/file.txtcontent'
    var_14 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00\x14\x00\x00\x00'
    var_15 = b'\x00\x00\x00\x00'
    var_16 = 'http://example.com/test.zip'
    var_17 = True
    var_18 = 'file.txt'
    var_19 = 'invalid.zip'
    var_20 = 'not a zip file'
    var_21 = False
    var_22 = 'empty.zip'
    var_23 = False
    var_24 = 'no_dir.zip'
    var_25 = 'file.txt'
    var_26 = 'content'
    var_27 = False
    var_28 = 'password.zip'
    var_29 = 'test_dir/'
    var_30 = ''
    var_31 = 'test_dir/file.txt'
    var_32 = 'content'
    var_33 = b'password'
    var_34 = False
    var_35 = 'password'
    var_36 = 'file.txt'
    var_37 = module_0.exists(var_13)
    var_38 = False
    var_39 = 'wrong'
    var_40 = False
    var_41 = True
    var_42 = False
    var_43 = 'file.txt'
    var_44 = module_0.exists(var_31)



# Parsed testcases at query #3
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
    var_11 = 'not a zip file'
    var_12 = False
    var_13 = 'empty.zip'
    var_14 = False
    var_15 = 'no_dir.zip'
    var_16 = 'file.txt'
    var_17 = 'content'
    var_18 = False
    var_19 = b'PK\x03\x04...'
    var_20 = 'http://example.com/protected.zip'
    var_21 = True
    var_22 = 'test_password'



# Parsed testcases at query #4
#--------------------------


import genericpath as module_0
import requests.cookies as module_1
import locale as module_2

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = 'test.zip'
    var_4 = 'file.txt'
    var_5 = 'test_repo/file.txt'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = b'fake zip content'
    var_9 = module_1.MockResponse(var_8)
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
    var_14 = module_2.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_2.str(var_17)
    var_21 = False
    var_22 = 'protected.zip'
    var_23 = var_20 / var_22
    var_24 = 'file.txt'
    var_25 = 'content'
    var_26 = b'password'
    var_27 = module_2.str(var_23)
    var_28 = False
    var_29 = 'password'
    var_30 = module_0.exists()
    var_31 = module_2.str(var_23)
    var_32 = False
    var_33 = 'wrong'
    var_34 = module_2.str(var_23)
    var_35 = False
    var_36 = True
    var_37 = 'invalid.zip'
    var_38 = var_34 / var_37
    var_39 = 'not a zip file'
    var_40 = module_2.str(var_38)
    var_41 = False



# Parsed testcases at query #5
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
    var_13 = 'empty.zip'
    var_14 = var_8 / var_13
    var_15 = module_1.str(var_14)
    var_16 = False
    var_17 = 'fake.zip'
    var_18 = var_15 / var_17
    var_19 = 'not a zip file'
    var_20 = module_1.str(var_18)
    var_21 = False
    var_22 = 'protected.zip'
    var_23 = var_20 / var_22
    var_24 = 'file.txt'
    var_25 = module_1.str(var_22)
    var_26 = 'test_repo/file.txt'
    var_27 = b'secret'
    var_28 = module_1.str(var_23)
    var_29 = False
    var_30 = 'secret'
    var_31 = module_0.exists()
    var_32 = module_1.str(var_23)
    var_33 = False
    var_34 = 'wrong'



# Parsed testcases at query #6
#--------------------------


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
    var_11 = 'bad.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'protected.zip'
    var_16 = 'test_dir/'
    var_17 = ''
    var_18 = 'test_dir/file.txt'
    var_19 = 'content'
    var_20 = b'password'
    var_21 = False
    var_22 = 'password'
    var_23 = False
    var_24 = 'wrong'
    var_25 = False
    var_26 = True
    var_27 = 'invalid.zip'
    var_28 = b'Not a zip file'
    var_29 = False



# Parsed testcases at query #7
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'test content'
    var_1 = 'test_dir/'
    var_2 = 'http://example.com/test.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3)
    var_5 = 'test_dir'
    var_6 = 'test_dir/'
    var_7 = '/path/to/test.zip'
    var_8 = False
    var_9 = module_0.unzip(var_7, var_8)
    var_10 = 'test_dir'
    var_11 = b'test content'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = module_0.unzip(var_12, var_13)
    var_15 = b'test content'
    var_16 = 'test_file.txt'
    var_17 = 'http://example.com/test.zip'
    var_18 = True
    var_19 = module_0.unzip(var_17, var_18)
    var_20 = b'test content'
    var_21 = 'test_dir/'
    var_22 = 'Password required'
    var_23 = 'http://example.com/test.zip'
    var_24 = True
    var_25 = 'correct_password'
    var_26 = module_0.unzip(var_23, var_24, password=var_25)
    var_27 = 'test_dir'
    var_28 = b'test content'
    var_29 = 'test_dir/'
    var_30 = 'Password required'
    var_31 = 'http://example.com/test.zip'
    var_32 = True
    var_33 = 'wrong_password'
    var_34 = module_0.unzip(var_31, var_32, password=var_33)
    var_35 = b'test content'
    var_36 = 'Bad zip file'
    var_37 = 'http://example.com/test.zip'
    var_38 = True
    var_39 = module_0.unzip(var_37, var_38)



# Parsed testcases at query #8
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
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_11 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00\x18\x00\x00\x00'
    var_12 = b'\x00\x00\x00\x00\x00\x00'
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
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
    var_32 = 'file.txt'
    var_33 = module_0.exists(var_29)
    var_34 = False
    var_35 = 'wrong'
    var_36 = False
    var_37 = True



# Parsed testcases at query #9
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
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = module_0.exists()
    var_12 = 'file.txt'
    var_13 = module_0.exists()
    var_14 = 'invalid.zip'
    var_15 = var_9 / var_14
    var_16 = 'not a zip file'
    var_17 = module_1.str(var_15)
    var_18 = False
    var_19 = 'empty.zip'
    var_20 = var_17 / var_19
    var_21 = module_1.str(var_20)
    var_22 = False
    var_23 = 'password.zip'
    var_24 = var_21 / var_23
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'content'
    var_29 = b'password'
    var_30 = module_1.str(var_24)
    var_31 = False
    var_32 = True
    var_33 = module_1.str(var_24)
    var_34 = False
    var_35 = 'password'
    var_36 = module_0.exists()



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_7 = b'content'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'bad.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
    var_15 = False
    var_16 = 'protected.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'content'
    var_21 = b'secret'
    var_22 = False
    var_23 = 'secret'
    var_24 = False
    var_25 = 'wrong'
    var_26 = False
    var_27 = True
    var_28 = 'invalid.zip'
    var_29 = b'not a zip file'
    var_30 = False



# Parsed testcases at query #11
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
    assert var_6 == 'test content'
    var_7 = module_0.exists()
    var_8 = 'empty.zip'
    var_9 = var_4 / var_8
    var_10 = module_1.str(var_9)
    var_11 = False
    var_12 = 'bad.zip'
    var_13 = var_10 / var_12
    var_14 = 'file.txt'
    var_15 = 'content'
    var_16 = module_1.str(var_13)
    var_17 = False
    var_18 = 'protected.zip'
    var_19 = var_16 / var_18
    var_20 = 'test_repo/file.txt'
    var_21 = 'content'
    var_22 = b'password'
    var_23 = module_1.str(var_19)
    var_24 = False
    var_25 = module_1.str(var_19)
    var_26 = False
    var_27 = 'password'
    var_28 = module_0.exists()
    var_29 = 'invalid.zip'
    var_30 = var_23 / var_29
    var_31 = 'not a zip file'
    var_32 = module_1.str(var_30)
    var_33 = False
    var_34 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_35 = b'test_repo/file.txttest content'
    var_36 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_37 = 'http://example.com/test.zip'
    var_38 = True
    var_39 = module_0.exists()
    var_40 = 'file.txt'



# Parsed testcases at query #12
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
    var_8 = None
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = 'empty.zip'
    var_12 = False
    var_13 = 'invalid.zip'
    var_14 = 'not a zip file'
    var_15 = False
    var_16 = 'password.zip'
    var_17 = 'test_dir/'
    var_18 = ''
    var_19 = 'test_dir/file.txt'
    var_20 = 'content'
    var_21 = b'secret'
    var_22 = False
    var_23 = 'secret'
    var_24 = False
    var_25 = 'wrong'
    var_26 = False
    var_27 = True



# Parsed testcases at query #13
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
    var_7 = module_0.exists(var_4)
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00test_dir/file.txt\x00\x00'
    var_11 = b'test content'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'no_dir.zip'
    var_21 = 'file.txt'
    var_22 = 'test content'
    var_23 = False
    var_24 = 'password.zip'
    var_25 = 'test_dir/'
    var_26 = ''
    var_27 = 'test_dir/file.txt'
    var_28 = 'test content'
    var_29 = b'password'
    var_30 = False
    var_31 = 'password'
    var_32 = 'file.txt'
    var_33 = module_0.exists(var_13)
    var_34 = False
    var_35 = 'wrong'
    var_36 = False
    var_37 = True



# Parsed testcases at query #14
#--------------------------


import cookiecutter.zipfile as module_0
import builtins as module_1

def test_case_0():
    var_0 = b'test data'
    var_1 = 'test_dir/'
    var_2 = 'http://example.com/test.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3, no_input=var_3)
    var_5 = 'test_dir'
    var_6 = 'local_dir/'
    var_7 = '/path/to/local.zip'
    var_8 = False
    var_9 = module_0.unzip(var_7, var_8)
    var_10 = 'local_dir'
    var_11 = b''
    var_12 = 'http://example.com/empty.zip'
    var_13 = True
    var_14 = module_0.unzip(var_12, var_13, no_input=var_13)
    var_15 = b'test data'
    var_16 = 'file.txt'
    var_17 = 'http://example.com/invalid.zip'
    var_18 = True
    var_19 = module_0.unzip(var_17, var_18, no_input=var_18)
    var_20 = b'test data'
    var_21 = 'protected_dir/'
    var_22 = module_1.RuntimeError()
    var_23 = None
    var_24 = 'http://example.com/protected.zip'
    var_25 = True
    var_26 = 'correct'
    var_27 = module_0.unzip(var_24, var_25, no_input=var_25, password=var_26)
    var_28 = 'protected_dir'
    var_29 = b'test data'
    var_30 = 'protected_dir/'
    var_31 = 'http://example.com/protected.zip'
    var_32 = True
    var_33 = 'incorrect'
    var_34 = module_0.unzip(var_31, var_32, no_input=var_32, password=var_33)
    var_35 = b'invalid data'
    var_36 = 'http://example.com/invalid.zip'
    var_37 = True
    var_38 = module_0.unzip(var_36, var_37, no_input=var_37)



# Parsed testcases at query #15
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
    var_10 = b'test_dir/file.txt\x00\x00\x00\x00\x00\x00\x00\x00'
    var_11 = b'test content'
    var_12 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00'
    var_13 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00'
    var_14 = b'\x18\x00\x00\x00\x00\x00\x00\x00'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = 'file.txt'
    var_18 = 'invalid.zip'
    var_19 = 'not a zip file'
    var_20 = False
    var_21 = 'empty.zip'
    var_22 = False
    var_23 = 'no_dir.zip'
    var_24 = 'file.txt'
    var_25 = 'test content'
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = 'test_dir/'
    var_29 = ''
    var_30 = 'test_dir/file.txt'
    var_31 = 'test content'
    var_32 = b'password'
    var_33 = False
    var_34 = 'password'
    var_35 = 'file.txt'
    var_36 = module_0.exists(var_32)
    var_37 = False
    var_38 = 'wrong'
    var_39 = False
    var_40 = True



# Parsed testcases at query #16
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
    var_8 = 'test_repo'
    var_9 = var_4 / var_8
    var_10 = 'file.txt'
    var_11 = var_9 / var_10
    var_12 = 'test content'
    var_13 = 'test.zip'
    assert var_13 == 'test content'
    var_14 = 'file.txt'
    var_15 = var_9 / var_14
    var_16 = 'test_repo/file.txt'
    var_17 = 'http://example.com/test.zip'
    var_18 = True
    var_19 = module_0.exists()
    var_20 = 'file.txt'
    var_21 = var_12 / var_20
    var_22 = 'test_repo'
    var_23 = var_14 / var_22
    var_24 = 'file.txt'
    var_25 = var_23 / var_24
    var_26 = 'test content'
    var_27 = 'test.zip'
    var_28 = var_21 / var_27
    var_29 = 'file.txt'
    var_30 = var_23 / var_29
    var_31 = 'test_repo/file.txt'
    var_32 = b'secret'
    var_33 = module_1.str(var_28)
    var_34 = False
    var_35 = 'secret'
    var_36 = module_0.exists()
    var_37 = 'invalid.zip'
    var_38 = var_29 / var_37
    var_39 = 'not a zip file'
    var_40 = module_1.str(var_38)
    var_41 = False
    var_42 = 'empty.zip'
    var_43 = var_40 / var_42
    var_44 = module_1.str(var_43)
    var_45 = False
    var_46 = 'test.txt'
    var_47 = var_44 / var_46
    var_48 = 'test content'
    var_49 = 'test.zip'
    var_50 = var_32 / var_49
    var_51 = 'test.txt'
    var_52 = module_1.str(var_50)
    var_53 = False



# Parsed testcases at query #17
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'fake content'
    var_1 = 'dir/'
    var_2 = 'dir/file.txt'
    var_3 = 'http://example.com/repo.zip'
    var_4 = True
    var_5 = module_0.unzip(var_3, var_4)
    var_6 = 'dir'
    var_7 = 100
    var_8 = 'dir/'
    var_9 = 'dir/file.txt'
    var_10 = '/path/to/local.zip'
    var_11 = False
    var_12 = module_0.unzip(var_10, var_11)
    var_13 = 'dir'
    var_14 = 'http://example.com/empty.zip'
    var_15 = True
    var_16 = module_0.unzip(var_14, var_15)
    var_17 = 'file.txt'
    var_18 = 'http://example.com/no_dir.zip'
    var_19 = True
    var_20 = module_0.unzip(var_18, var_19)
    var_21 = 'dir/'
    var_22 = 'dir/file.txt'
    var_23 = 'Password required'
    var_24 = 'http://example.com/protected.zip'
    var_25 = True
    var_26 = 'correct_password'
    var_27 = module_0.unzip(var_24, var_25, password=var_26)
    var_28 = 'dir'
    var_29 = 'dir/'
    var_30 = 'dir/file.txt'
    var_31 = 'Password required'
    var_32 = 'http://example.com/protected.zip'
    var_33 = True
    var_34 = 'wrong_password'
    var_35 = module_0.unzip(var_32, var_33, password=var_34)
    var_36 = 'Bad zip file'
    var_37 = 'http://example.com/bad.zip'
    var_38 = True
    var_39 = module_0.unzip(var_37, var_38)



# Parsed testcases at query #18
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
    var_16 = 'not a zip file'
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



# Parsed testcases at query #19
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
    var_7 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00'
    var_8 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_9 = b'\x01\x00\x00\x00test_dir/\x00PK\x01\x02\x14\x00'
    var_10 = b'\x14\x00\x00\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00'
    var_11 = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_12 = b'test_dir/PK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00'
    var_13 = b'\x00\x00\x00\x00'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = 'invalid.zip'
    var_17 = b'not a zip file'
    var_18 = False
    var_19 = 'empty.zip'
    var_20 = False
    var_21 = 'no_dir.zip'
    var_22 = 'file.txt'
    var_23 = 'content'
    var_24 = False
    var_25 = 'password.zip'
    var_26 = 'test_dir/'
    var_27 = ''
    var_28 = 'test_dir/file.txt'
    var_29 = 'content'
    var_30 = b'password'
    var_31 = False
    var_32 = False
    var_33 = 'password'
    var_34 = 'file.txt'
    var_35 = module_0.exists(var_13)



# Parsed testcases at query #20
#--------------------------


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
    var_11 = 'non_zip.txt'
    var_12 = 'not a zip file'
    var_13 = False
    var_14 = 'protected.zip'
    var_15 = 'test_dir/'
    var_16 = ''
    var_17 = 'test_dir/file.txt'
    var_18 = 'content'
    var_19 = b'password'
    var_20 = False
    var_21 = 'password'
    var_22 = False
    var_23 = 'wrong'
    var_24 = False
    var_25 = True



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'invalid.zip'
    var_7 = 'not a zip file'
    var_8 = False
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'no_dir.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'protected.zip'
    var_16 = 'test_dir/'
    var_17 = ''
    var_18 = 'test_dir/file.txt'
    var_19 = 'content'
    var_20 = b'password'
    var_21 = False
    var_22 = True
    var_23 = 'protected.zip'
    var_24 = 'test_dir/'
    var_25 = ''
    var_26 = 'test_dir/file.txt'
    var_27 = 'content'
    var_28 = b'password'
    var_29 = False
    var_30 = 'password'



# Parsed testcases at query #22
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
    var_8 = b'PK\x03\x04\x14\x00\x00\x00\x08\x00'
    var_9 = b'test_dir/\x00\x00\x00\x00\x00\x00\x00\x00'
    var_10 = b'\x00\x00\x00\x00\x00\x00\x00\x00test_dir/file.txtcontent'
    var_11 = b'PK\x01\x02\x14\x00\x14\x00\x00\x00\x00\x00'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'file.txt'
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = True
    var_19 = 'empty.zip'
    var_20 = False
    var_21 = True
    var_22 = 'no_dir.zip'
    var_23 = 'file.txt'
    var_24 = 'content'
    var_25 = False
    var_26 = True
    var_27 = 'password.zip'
    var_28 = 'test_dir/'
    var_29 = ''
    var_30 = 'test_dir/file.txt'
    var_31 = 'content'
    var_32 = False
    var_33 = True
    var_34 = 'test_password'
    var_35 = 'file.txt'
    var_36 = 'password.zip'
    var_37 = 'test_dir/'
    var_38 = ''
    var_39 = 'test_dir/file.txt'
    var_40 = 'content'
    var_41 = False
    var_42 = True
    var_43 = 'wrong_password'



# Parsed testcases at query #23
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
    var_7 = module_0.exists(var_4)
    var_8 = b'PK\x03\x04'
    var_9 = b'test content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'invalid.zip'
    var_13 = b'not a zip file'
    var_14 = False
    var_15 = 'empty.zip'
    var_16 = False
    var_17 = 'no_dir.zip'
    var_18 = 'file.txt'
    var_19 = 'test content'
    var_20 = False
    var_21 = 'password.zip'
    var_22 = 'test_dir/'
    var_23 = ''
    var_24 = 'test_dir/file.txt'
    var_25 = 'test content'
    var_26 = b'password'
    var_27 = False
    var_28 = 'password'
    var_29 = False
    var_30 = 'wrong'
    var_31 = False
    var_32 = True



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = 'http://example.com/test.zip'
    var_8 = True
    var_9 = 'file.txt'
    var_10 = 'invalid.zip'
    var_11 = 'not a zip file'
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
    var_27 = 'file.txt'
    var_28 = False
    var_29 = 'wrong'
    var_30 = False
    var_31 = True



# Parsed testcases at query #25
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
    var_9 = b'test_dir/\x00\x00\x00\x00\x00\x00'
    var_10 = b'test_dir/file.txtcontent'
    var_11 = b'PK\x05\x06\x00\x00\x00\x00\x01\x00'
    var_12 = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    var_13 = b'\x00\x00\x00\x00\x00\x00'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = 'file.txt'
    var_17 = 'protected.zip'
    var_18 = 'test_dir/'
    var_19 = ''
    var_20 = 'test_dir/file.txt'
    var_21 = 'content'
    var_22 = b'secret'
    var_23 = False
    var_24 = 'secret'
    var_25 = False
    var_26 = 'wrong'
    var_27 = 'invalid.zip'
    var_28 = 'not a zip file'
    var_29 = False
    var_30 = 'empty.zip'
    var_31 = False
    var_32 = 'no_dir.zip'
    var_33 = 'file.txt'
    var_34 = 'content'
    var_35 = False



