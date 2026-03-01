####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = 'not a zip file'
    var_10 = module_1.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_1.str(var_17)
    var_21 = False
    var_22 = b'chunk1'
    var_23 = b'chunk2'
    var_24 = None
    var_25 = b'PK\x03\x04'
    var_26 = b' '
    var_27 = 100
    var_28 = var_26 * var_27
    var_29 = var_25 + var_28
    var_30 = 'test_project/'
    var_31 = 'test_project/file.txt'
    var_32 = 'http://example.com/test.zip'
    var_33 = True
    var_34 = 100
    var_35 = 'protected.zip'
    var_36 = var_32 / var_35
    var_37 = 'test_project/'
    var_38 = ''
    var_39 = b'secret'
    var_40 = b'secret'
    var_41 = 'test_project/file.txt'
    var_42 = 'protected content'
    var_43 = module_1.str(var_36)
    var_44 = False
    var_45 = None
    var_46 = 'protected2.zip'
    var_47 = var_43 / var_46
    var_48 = 'test_project/'
    var_49 = ''
    var_50 = b'secret'
    var_51 = b'secret'
    var_52 = 'test_project/file.txt'
    var_53 = 'content'
    var_54 = module_1.str(var_47)
    var_55 = False
    var_56 = True
    var_57 = None
    var_58 = 'protected3.zip'
    var_59 = var_54 / var_58
    var_60 = 'test_project/'
    var_61 = ''
    var_62 = b'secret'
    var_63 = b'secret'
    var_64 = 'test_project/file.txt'
    var_65 = 'content'
    var_66 = module_1.str(var_59)
    var_67 = False
    var_68 = None
    var_69 = b'data'
    var_70 = 'project/'
    var_71 = 'project/file.txt'
    var_72 = 'http://example.com/existing.zip'
    var_73 = True
    var_74 = False
    var_75 = 'test/'
    var_76 = 'test/file.txt'
    var_77 = 'http://example.com/test.zip'
    var_78 = True
    var_79 = '~/some/path'
    var_80 = module_2.unzip(var_77, var_78, var_79, var_78)



# Parsed testcases at query #2
#--------------------------


import locale as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = b'mock zip content'
    var_8 = None
    var_9 = 'project/'
    var_10 = 'project/file.txt'
    var_11 = None
    var_12 = 'http://example.com/repo.zip'
    var_13 = True
    var_14 = 'empty.zip'
    var_15 = var_12 / var_14
    var_16 = module_0.str(var_15)
    var_17 = False
    var_18 = 'flat.zip'
    var_19 = var_16 / var_18
    var_20 = 'file.txt'
    var_21 = 'content'
    var_22 = module_0.str(var_19)
    var_23 = False
    var_24 = 'protected.zip'
    var_25 = var_22 / var_24
    var_26 = 'project/'
    var_27 = ''
    var_28 = b'secret'
    var_29 = 'project/'
    var_30 = 'project/file.txt'
    var_31 = None
    var_32 = 'password required'
    var_33 = module_0.str(var_25)
    var_34 = False
    var_35 = 'secret'
    var_36 = 'invalid.zip'
    var_37 = var_29 / var_36
    var_38 = b'not a zip file'
    var_39 = module_0.str(var_37)
    var_40 = False
    var_41 = 'new'
    var_42 = var_39 / var_41
    var_43 = 'nested'
    var_44 = var_42 / var_43
    var_45 = 'dir'
    var_46 = var_44 / var_45
    var_47 = 'test.zip'
    var_48 = var_34 / var_47
    var_49 = 'project/'
    var_50 = ''
    var_51 = module_0.str(var_48)
    var_52 = False
    var_53 = module_1.unzip(var_51, var_52, var_46)
    var_54 = module_2.exists()
    var_55 = b'new content'
    var_56 = None
    var_57 = 'repo.zip'
    var_58 = var_42 / var_57
    var_59 = b'old content'
    var_60 = 'project/'
    var_61 = None
    var_62 = 'http://example.com/repo.zip'
    var_63 = True



# Parsed testcases at query #3
#--------------------------


import genericpath as module_0
import locale as module_1
import _io as module_2
import cookiecutter.zipfile as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = b'not a zip file'
    var_10 = module_1.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
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
    var_24 = 'project/'
    var_25 = ''
    var_26 = 'project/file.txt'
    var_27 = 'content'
    var_28 = b'secret'
    var_29 = module_1.str(var_23)
    var_30 = False
    var_31 = 'secret'
    var_32 = 'protected.zip'
    var_33 = var_24 / var_32
    var_34 = 'project/'
    var_35 = ''
    var_36 = 'project/file.txt'
    var_37 = 'content'
    var_38 = b'secret'
    var_39 = module_1.str(var_33)
    var_40 = False
    var_41 = 'wrong'
    var_42 = b'chunk1'
    var_43 = b'chunk2'
    var_44 = None
    var_45 = module_2.BytesIO()
    var_46 = 'url_project/'
    var_47 = ''
    var_48 = 'url_project/file.txt'
    var_49 = 'content'
    var_50 = 'http://example.com/repo.zip'
    var_51 = True
    var_52 = 100
    var_53 = 'http://example.com/existing.zip'
    var_54 = True
    var_55 = False
    var_56 = 'protected.zip'
    var_57 = var_53 / var_56
    var_58 = 'project/'
    var_59 = ''
    var_60 = 'project/file.txt'
    var_61 = 'content'
    var_62 = b'secret'
    var_63 = module_1.str(var_57)
    var_64 = False
    var_65 = True
    var_66 = 'test.zip'
    var_67 = var_63 / var_66
    var_68 = 'tilde_project/'
    var_69 = ''
    var_70 = 'tilde_project/file.txt'
    var_71 = 'content'
    var_72 = module_1.str(var_67)
    var_73 = False
    var_74 = '~/some/path'
    var_75 = module_3.unzip(var_72, var_73, var_74)



# Parsed testcases at query #4
#--------------------------


import locale as module_0
import _io as module_1
import cookiecutter.zipfile as module_2
import genericpath as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = 'file.txt'
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = 'not a zip file'
    var_10 = module_0.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
    var_14 = module_0.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_0.str(var_17)
    var_21 = False
    var_22 = 'protected.zip'
    var_23 = var_20 / var_22
    var_24 = 'project/'
    var_25 = ''
    var_26 = 'project/secret.txt'
    var_27 = 'confidential'
    var_28 = b'secret'
    var_29 = module_0.str(var_23)
    var_30 = False
    var_31 = 'secret'
    var_32 = 'protected.zip'
    var_33 = var_24 / var_32
    var_34 = 'project/'
    var_35 = ''
    var_36 = 'project/secret.txt'
    var_37 = 'confidential'
    var_38 = b'secret'
    var_39 = module_0.str(var_33)
    var_40 = False
    var_41 = 'wrong'
    var_42 = b'chunk1'
    var_43 = b'chunk2'
    var_44 = None
    var_45 = module_1.BytesIO()
    var_46 = 'webproject/'
    var_47 = ''
    var_48 = 'webproject/index.html'
    var_49 = '<html></html>'
    var_50 = 'webproject/'
    var_51 = 'webproject/index.html'
    var_52 = 'http://example.com/repo.zip'
    var_53 = True
    var_54 = 100
    var_55 = 'protected.zip'
    var_56 = var_52 / var_55
    var_57 = 'project/'
    var_58 = ''
    var_59 = 'project/file.txt'
    var_60 = 'content'
    var_61 = b'secret'
    var_62 = module_0.str(var_56)
    var_63 = False
    var_64 = True
    var_65 = 'newdir'
    var_66 = var_62 / var_65
    var_67 = 'test.zip'
    var_68 = var_64 / var_67
    var_69 = 'project/'
    var_70 = ''
    var_71 = 'project/test.txt'
    var_72 = 'test'
    var_73 = module_0.str(var_68)
    var_74 = False
    var_75 = module_0.str(var_66)
    var_76 = module_2.unzip(var_73, var_74, var_75)
    var_77 = module_3.exists()
    var_78 = 'relative.zip'
    var_79 = var_70 / var_78
    var_80 = 'relativeproject/'
    var_81 = ''
    var_82 = 'relativeproject/file.txt'
    var_83 = 'data'
    var_84 = module_0.str(var_79)
    var_85 = False
    var_86 = '.'
    var_87 = module_2.unzip(var_84, var_85, var_86)
    var_88 = 'file.txt'
    var_89 = module_3.exists(var_77)



# Parsed testcases at query #5
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'file.txt'
    var_8 = module_0.exists()

def test_case_0():
    var_0 = b'mock zip content'
    var_1 = b'PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00project/\x00\x00\x00'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'http://example.com/test.zip'
    var_5 = True

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False

def test_case_0():
    var_0 = 'flat.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = b'secret'
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = 'password required'
    var_7 = None
    var_8 = False
    var_9 = 'secret'

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = b'cached content'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'http://example.com/test.zip'
    var_5 = True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = b'cached content'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'http://example.com/test.zip'
    var_5 = True



# Parsed testcases at query #6
#--------------------------


import genericpath as module_0
import locale as module_1
import builtins as module_2
import cookiecutter.zipfile as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'mock_zip_content'
    var_8 = 'test_project/'
    var_9 = 'test_project/file.txt'
    var_10 = 'http://example.com/repo.zip'
    var_11 = True
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = module_1.str(var_10)
    var_17 = 'bad.zip'
    var_18 = var_14 / var_17
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = module_1.str(var_18)
    var_22 = False
    var_23 = module_1.str(var_10)
    var_24 = 'protected.zip'
    var_25 = var_21 / var_24
    var_26 = 'test_project/'
    var_27 = 'test_project/file.txt'
    var_28 = module_2.RuntimeError()
    var_29 = None
    var_30 = module_1.str(var_25)
    var_31 = False
    var_32 = 'secret'
    var_33 = 'invalid.zip'
    var_34 = var_26 / var_33
    var_35 = 'not a zip file'
    var_36 = module_1.str(var_34)
    var_37 = False
    var_38 = module_1.str(var_30)
    var_39 = 'repo.zip'
    var_40 = var_36 / var_39
    var_41 = b'existing content'
    var_42 = b'new_content'
    var_43 = 'project/'
    var_44 = 'project/file.txt'
    var_45 = 'http://example.com/repo.zip'
    var_46 = True
    var_47 = 'new_directory'
    var_48 = var_43 / var_47
    var_49 = 'test.zip'
    var_50 = var_45 / var_49
    var_51 = 'project/'
    var_52 = ''
    var_53 = 'project/file.txt'
    var_54 = 'content'
    var_55 = module_1.str(var_50)
    var_56 = False
    var_57 = module_3.unzip(var_55, var_56, var_48)
    var_58 = module_0.exists()



# Parsed testcases at query #7
#--------------------------


import genericpath as module_0
import locale as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = 'not a zip file'
    var_10 = module_1.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = var_14 / var_16
    var_18 = 'file.txt'
    var_19 = 'content'
    var_20 = module_1.str(var_17)
    var_21 = False
    var_22 = b'chunk1'
    var_23 = b'chunk2'
    var_24 = 'project/'
    var_25 = 'project/file.txt'
    var_26 = 'http://example.com/repo.zip'
    var_27 = True
    var_28 = 100
    var_29 = 'protected.zip'
    var_30 = var_24 / var_29
    var_31 = b'secret'
    var_32 = 'project/'
    var_33 = ''
    var_34 = 'project/file.txt'
    var_35 = 'content'
    var_36 = module_1.str(var_30)
    var_37 = False
    var_38 = 'secret'
    var_39 = 'protected.zip'
    var_40 = var_32 / var_39
    var_41 = 'project/'
    var_42 = 'project/file.txt'
    var_43 = 'password required'
    var_44 = 'bad password'
    var_45 = module_1.str(var_40)
    var_46 = False
    var_47 = 'wrong'
    var_48 = 'project/'
    var_49 = 'project/file.txt'
    var_50 = 'password required'
    var_51 = 'test.zip'
    var_52 = False
    var_53 = True
    var_54 = b'chunk1'
    var_55 = 'project/'
    var_56 = 'project/file.txt'
    var_57 = 'http://example.com/repo.zip'
    var_58 = True
    var_59 = 'project/'
    var_60 = 'project/file.txt'
    var_61 = 'http://example.com/repo.zip'
    var_62 = True



# Parsed testcases at query #8
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()

def test_case_0():
    var_0 = b'mock zip content'
    var_1 = 'test_project/'
    var_2 = 'test_project/file.txt'
    var_3 = 'http://example.com/test.zip'
    var_4 = True

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False

def test_case_0():
    var_0 = 'bad.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = b'secret'
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'secret'

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = b'existing cache'
    var_2 = b'new content'
    var_3 = 'test_project/'
    var_4 = 'http://example.com/test.zip'
    var_5 = True

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = b'secret'
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = b'secret'
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False



# Parsed testcases at query #9
#--------------------------


import cookiecutter.zipfile as module_0
import locale as module_1
import zipfile as module_2
import posixpath as module_3

def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'http://example.com/repo.zip'
    var_5 = True
    var_6 = '.'
    var_7 = module_0.unzip(var_4, var_5, var_6)
    assert var_7 == '/tmp/tempdir/project'
    var_8 = 100
    var_9 = '/tmp/tempdir'
    var_10 = 'project/'
    var_11 = 'project/file.txt'
    var_12 = '/local/repo.zip'
    var_13 = False
    var_14 = '.'
    var_15 = module_0.unzip(var_12, var_13, var_14)
    assert var_15 == '/tmp/tempdir/project'
    var_16 = '/tmp/tempdir'
    var_17 = 'http://example.com/repo.zip'
    var_18 = True
    var_19 = module_0.unzip(var_17, var_18)
    var_20 = module_1.str(var_17)
    var_21 = module_1.str(var_19)
    var_22 = 'file.txt'
    var_23 = 'http://example.com/repo.zip'
    var_24 = True
    var_25 = module_0.unzip(var_23, var_24)
    var_26 = module_1.str(var_24)
    var_27 = module_1.str(var_21)
    var_28 = 'project/'
    var_29 = 'project/file.txt'
    var_30 = 'http://example.com/repo.zip'
    var_31 = True
    var_32 = 'secret'
    var_33 = module_0.unzip(var_30, var_31, password=var_32)
    assert var_33 == '/tmp/tempdir/project'
    var_34 = '/tmp/tempdir'
    var_35 = b'secret'
    var_36 = 'http://example.com/repo.zip'
    var_37 = True
    var_38 = module_0.unzip(var_36, var_37)
    var_39 = module_1.str(var_36)
    var_40 = module_1.str(var_38)
    var_41 = b'chunk1'
    var_42 = 'project/'
    var_43 = 'project/file.txt'
    var_44 = 'http://example.com/repo.zip'
    var_45 = True
    var_46 = module_0.unzip(var_44, var_45, no_input=var_45)
    assert var_46 == '/tmp/tempdir/project'
    var_47 = 'project/'
    var_48 = 'project/file.txt'
    var_49 = '/custom/dir'
    var_50 = module_2.Path(var_49)
    var_51 = 'http://example.com/repo.zip'
    var_52 = True
    var_53 = module_0.unzip(var_51, var_52, var_50)
    assert var_53 == '/tmp/tempdir/project'
    var_54 = module_3.expanduser()



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import genericpath as module_0
import locale as module_1
import _io as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = 'not a zip file'
    var_10 = module_1.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
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
    var_24 = 'project/'
    var_25 = ''
    var_26 = 'project/file.txt'
    var_27 = 'content'
    var_28 = b'secret'
    var_29 = module_1.str(var_23)
    var_30 = False
    var_31 = 'protected.zip'
    var_32 = var_29 / var_31
    var_33 = 'project/'
    var_34 = ''
    var_35 = 'project/file.txt'
    var_36 = 'content'
    var_37 = b'secret'
    var_38 = module_1.str(var_32)
    var_39 = False
    var_40 = True
    var_41 = b'zip content'
    var_42 = module_2.BytesIO()
    var_43 = 'project/'
    var_44 = ''
    var_45 = 'project/file.txt'
    var_46 = 'content'
    var_47 = 0
    var_48 = 'http://example.com/test.zip'
    var_49 = True
    var_50 = 'protected.zip'
    var_51 = var_48 / var_50
    var_52 = 'project/'
    var_53 = ''
    var_54 = 'project/file.txt'
    var_55 = 'content'
    var_56 = b'correct'
    var_57 = module_1.str(var_51)
    var_58 = False
    var_59 = 'protected.zip'
    var_60 = var_57 / var_59
    var_61 = 'project/'
    var_62 = ''
    var_63 = 'project/file.txt'
    var_64 = 'content'
    var_65 = b'mypassword'
    var_66 = module_1.str(var_60)
    var_67 = False
    var_68 = 'mypassword'
    var_69 = 'protected.zip'
    var_70 = var_61 / var_69
    var_71 = 'project/'
    var_72 = ''
    var_73 = 'project/file.txt'
    var_74 = 'content'
    var_75 = b'mypassword'
    var_76 = module_1.str(var_70)
    var_77 = False
    var_78 = 'wrong'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.zipfile as module_0
import locale as module_1

def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'http://example.com/repo.zip'
    var_5 = True
    var_6 = '.'
    var_7 = module_0.unzip(var_4, var_5, var_6)
    assert var_7 == '/tmp/tempdir/project'
    var_8 = 100
    var_9 = 'project/'
    var_10 = 'project/file.txt'
    var_11 = '/local/repo.zip'
    var_12 = False
    var_13 = '.'
    var_14 = module_0.unzip(var_11, var_12, var_13)
    assert var_14 == '/tmp/tempdir/project'
    var_15 = 'http://example.com/repo.zip'
    var_16 = True
    var_17 = module_0.unzip(var_15, var_16)
    var_18 = 'Zip repository'
    var_19 = module_1.str(var_16)
    var_20 = var_18 in var_19
    var_21 = 'empty'
    var_22 = module_1.str(var_5)
    var_23 = var_21 in var_22
    var_24 = 'file.txt'
    var_25 = 'http://example.com/repo.zip'
    var_26 = True
    var_27 = module_0.unzip(var_25, var_26)
    var_28 = module_1.str(var_26)
    var_29 = 'project/'
    var_30 = 'project/file.txt'
    var_31 = 'password required'
    var_32 = None
    var_33 = 'http://example.com/repo.zip'
    var_34 = True
    var_35 = 'secret'
    var_36 = module_0.unzip(var_33, var_34, password=var_35)
    assert var_36 == '/tmp/tempdir/project'
    var_37 = '/tmp/tempdir'
    var_38 = b'secret'
    var_39 = 'project/'
    var_40 = 'password required'
    var_41 = 'http://example.com/repo.zip'
    var_42 = True
    var_43 = 'wrong'
    var_44 = module_0.unzip(var_41, var_42, no_input=var_42, password=var_43)
    var_45 = module_1.str(var_43)
    var_46 = 'http://example.com/repo.zip'
    var_47 = True
    var_48 = module_0.unzip(var_46, var_47)
    var_49 = module_1.str(var_46)
    var_50 = b'chunk1'
    var_51 = 'project/'
    var_52 = 'project/file.txt'
    var_53 = 'http://example.com/repo.zip'
    var_54 = True
    var_55 = False
    var_56 = module_0.unzip(var_53, var_54, no_input=var_55)
    assert var_56 == '/tmp/tempdir/project'
    var_57 = 'project/'
    var_58 = 'project/file.txt'
    var_59 = 'http://example.com/repo.zip'
    var_60 = True
    var_61 = False
    var_62 = module_0.unzip(var_59, var_60, no_input=var_61)
    assert var_62 == '/tmp/tempdir/project'
    var_63 = 'project/'
    var_64 = 'password required'
    var_65 = 'wrong password'
    var_66 = None
    var_67 = 'wrong1'
    var_68 = 'wrong2'
    var_69 = 'correct'
    var_70 = 'http://example.com/repo.zip'
    var_71 = True
    var_72 = False
    var_73 = module_0.unzip(var_70, var_71, no_input=var_72)
    assert var_73 == '/tmp/tempdir/project'



# Parsed testcases at query #3
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = 'invalid.zip'
    var_8 = var_1 / var_7
    var_9 = 'not a zip file'
    var_10 = module_1.str(var_8)
    var_11 = False
    var_12 = 'empty.zip'
    var_13 = var_10 / var_12
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
    var_24 = 'project/'
    var_25 = ''
    var_26 = 'project/file.txt'
    var_27 = 'content'
    var_28 = b'secret'
    var_29 = module_1.str(var_23)
    var_30 = False
    var_31 = 'secret'
    var_32 = 'protected.zip'
    var_33 = var_24 / var_32
    var_34 = 'project/'
    var_35 = ''
    var_36 = 'project/file.txt'
    var_37 = 'content'
    var_38 = b'secret'
    var_39 = module_1.str(var_33)
    var_40 = False
    var_41 = 'wrong'
    var_42 = b'chunk1'
    var_43 = b'chunk2'
    var_44 = None
    var_45 = 'project/'
    var_46 = 'project/file.txt'
    var_47 = 'http://example.com/repo.zip'
    var_48 = True
    var_49 = 'http://example.com/repo.zip'
    var_50 = True
    var_51 = 100
    var_52 = 'project/'
    var_53 = 'project/file.txt'
    var_54 = 'http://example.com/repo.zip'
    var_55 = True
    var_56 = 'custom'
    var_57 = var_52 / var_56
    var_58 = 'test.zip'
    var_59 = var_54 / var_58
    var_60 = 'project/'
    var_61 = ''
    var_62 = 'project/file.txt'
    var_63 = 'content'
    var_64 = module_1.str(var_59)
    var_65 = False
    var_66 = module_1.str(var_57)
    var_67 = module_2.unzip(var_64, var_65, var_66)
    var_68 = module_0.exists()
    var_69 = 'protected.zip'
    var_70 = var_60 / var_69
    var_71 = 'project/'
    var_72 = ''
    var_73 = 'project/file.txt'
    var_74 = 'content'
    var_75 = b'secret'
    var_76 = module_1.str(var_70)
    var_77 = False
    var_78 = True



# Parsed testcases at query #4
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'mock zip content'
    var_8 = 'test_project/'
    var_9 = 'test_project/file.txt'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
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
    var_22 = 'corrupt.zip'
    var_23 = var_20 / var_22
    var_24 = 'not a zip file'
    var_25 = module_1.str(var_23)
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = var_25 / var_27
    var_29 = 'test_project/'
    var_30 = 'test_project/file.txt'
    var_31 = 'password required'
    var_32 = None
    var_33 = module_1.str(var_28)
    var_34 = False
    var_35 = 'secret'
    var_36 = 'protected.zip'
    var_37 = var_29 / var_36
    var_38 = 'test_project/'
    var_39 = 'test_project/file.txt'
    var_40 = 'password required'
    var_41 = module_1.str(var_37)
    var_42 = False
    var_43 = True
    var_44 = b'mock zip content'
    var_45 = 'test_project/'
    var_46 = 'test_project/file.txt'
    var_47 = 'test.zip'
    var_48 = var_43 / var_47
    var_49 = 'existing content'
    var_50 = 'http://example.com/test.zip'
    var_51 = True
    var_52 = False
    var_53 = 'test.zip'
    var_54 = var_50 / var_53
    var_55 = 'test_project/'
    var_56 = ''
    var_57 = 'test_project/file.txt'
    var_58 = 'content'
    var_59 = module_1.str(var_54)
    var_60 = False
    var_61 = module_2.unzip(var_59, var_60, var_58)
    var_62 = module_1.str(var_54)
    var_63 = False
    var_64 = '~/test'
    var_65 = module_2.unzip(var_62, var_63, var_64)



# Parsed testcases at query #5
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'fake_zip_content'
    var_8 = None
    var_9 = None
    var_10 = 'test_project/'
    var_11 = 'test_project/file.txt'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = 'empty.zip'
    var_15 = var_9 / var_14
    var_16 = module_1.str(var_15)
    var_17 = False
    var_18 = 'bad.zip'
    var_19 = var_16 / var_18
    var_20 = 'file.txt'
    var_21 = 'content'
    var_22 = module_1.str(var_19)
    var_23 = False
    var_24 = 'invalid.zip'
    var_25 = var_22 / var_24
    var_26 = 'not a zip file'
    var_27 = module_1.str(var_25)
    var_28 = False
    var_29 = 'protected.zip'
    var_30 = var_27 / var_29
    var_31 = 'test_project/'
    var_32 = ''
    var_33 = b'secret'
    var_34 = 'test_project/file.txt'
    var_35 = 'content'
    var_36 = b'secret'
    var_37 = None
    var_38 = 'test_project/'
    var_39 = 'test_project/file.txt'
    var_40 = 'password required'
    var_41 = module_1.str(var_30)
    var_42 = False
    var_43 = 'secret'
    var_44 = 'protected.zip'
    var_45 = var_37 / var_44
    var_46 = None
    var_47 = 'test_project/'
    var_48 = 'test_project/file.txt'
    var_49 = 'password required'
    var_50 = module_1.str(var_45)
    var_51 = False
    var_52 = 'wrong'
    var_53 = True
    var_54 = b'new_content'
    var_55 = None
    var_56 = None
    var_57 = 'project/'
    var_58 = 'project/file.txt'
    var_59 = 'http://example.com/test.zip'
    var_60 = True
    var_61 = 'new_directory'
    var_62 = var_56 / var_61
    var_63 = None
    var_64 = 'project/'
    var_65 = 'project/file.txt'
    var_66 = 'test.zip'
    var_67 = var_59 / var_66
    var_68 = 'project/'
    var_69 = ''
    var_70 = module_1.str(var_67)
    var_71 = False
    var_72 = module_1.str(var_62)
    var_73 = module_2.unzip(var_70, var_71, var_72)
    var_74 = module_0.exists()



# Parsed testcases at query #6
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'mock zip content'
    var_8 = 'project/'
    var_9 = 'project/file.txt'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = module_1.str(var_10)
    var_17 = 'bad.zip'
    var_18 = var_14 / var_17
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = module_1.str(var_18)
    var_22 = False
    var_23 = module_1.str(var_10)
    var_24 = 'protected.zip'
    var_25 = var_21 / var_24
    var_26 = 'project/'
    var_27 = ''
    var_28 = b'secret'
    var_29 = 'project/'
    var_30 = 'project/file.txt'
    var_31 = 'password required'
    var_32 = None
    var_33 = module_1.str(var_25)
    var_34 = False
    var_35 = 'secret'
    var_36 = 'invalid.zip'
    var_37 = var_29 / var_36
    var_38 = 'not a zip file'
    var_39 = module_1.str(var_37)
    var_40 = False
    var_41 = module_1.str(var_32)
    var_42 = 'cached.zip'
    var_43 = var_39 / var_42
    var_44 = 'cached content'
    var_45 = b'new content'
    var_46 = 'project/'
    var_47 = 'project/file.txt'
    var_48 = 'http://example.com/cached.zip'
    var_49 = True
    var_50 = 'custom'
    var_51 = var_46 / var_50
    var_52 = 'test.zip'
    var_53 = var_48 / var_52
    var_54 = 'project/'
    var_55 = ''
    var_56 = 'project/file.txt'
    var_57 = 'content'
    var_58 = module_1.str(var_53)
    var_59 = False
    var_60 = module_2.unzip(var_58, var_59, var_51)
    var_61 = module_0.exists()



# Parsed testcases at query #7
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'mock zip content'
    var_8 = 'project/'
    var_9 = 'project/file.txt'
    var_10 = 'http://example.com/repo.zip'
    var_11 = True
    var_12 = 'empty.zip'
    var_13 = var_8 / var_12
    var_14 = module_1.str(var_13)
    var_15 = False
    var_16 = 'flat.zip'
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
    var_29 = 'project/'
    var_30 = 'project/file.txt'
    var_31 = 'password required'
    var_32 = None
    var_33 = module_1.str(var_28)
    var_34 = False
    var_35 = 'secret'
    var_36 = 'protected.zip'
    var_37 = var_29 / var_36
    var_38 = 'project/'
    var_39 = 'project/file.txt'
    var_40 = 'password required'
    var_41 = module_1.str(var_37)
    var_42 = False
    var_43 = True
    var_44 = 'clone'
    var_45 = var_41 / var_44
    var_46 = 'repo.zip'
    var_47 = var_45 / var_46
    var_48 = b'old content'
    var_49 = b'new content'
    var_50 = 'project/'
    var_51 = 'project/file.txt'
    var_52 = 'http://example.com/repo.zip'
    var_53 = True
    var_54 = module_1.str(var_45)
    var_55 = module_2.unzip(var_52, var_53, var_54, var_53)
    var_56 = 'test.zip'
    var_57 = 'project/'
    var_58 = ''
    var_59 = 'project/file.txt'
    var_60 = 'content'
    var_61 = False
    var_62 = '~'
    var_63 = module_2.unzip(var_57, var_61, var_62)
    var_64 = True



# Parsed testcases at query #8
#--------------------------


import genericpath as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.exists()
    var_7 = b'mock zip content'
    var_8 = 'project/'
    var_9 = 'project/file.txt'
    var_10 = 'http://example.com/repo.zip'
    var_11 = True
    var_12 = 100
    var_13 = 'empty.zip'
    var_14 = var_8 / var_13
    var_15 = module_1.str(var_14)
    var_16 = False
    var_17 = 'flat.zip'
    var_18 = var_15 / var_17
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = module_1.str(var_18)
    var_22 = False
    var_23 = 'invalid.zip'
    var_24 = var_21 / var_23
    var_25 = 'not a zip file'
    var_26 = module_1.str(var_24)
    var_27 = False
    var_28 = 'protected.zip'
    var_29 = var_26 / var_28
    var_30 = 'project/'
    var_31 = ''
    var_32 = b'secret'
    var_33 = 'project/'
    var_34 = 'project/file.txt'
    var_35 = 'password required'
    var_36 = None
    var_37 = module_1.str(var_29)
    var_38 = False
    var_39 = 'secret'
    var_40 = 'protected.zip'
    var_41 = var_33 / var_40
    var_42 = 'project/'
    var_43 = 'project/file.txt'
    var_44 = 'password required'
    var_45 = module_1.str(var_41)
    var_46 = False
    var_47 = True
    var_48 = b'mock zip content'
    var_49 = 'repo.zip'
    var_50 = var_46 / var_49
    var_51 = 'old content'
    var_52 = 'project/'
    var_53 = 'project/file.txt'
    var_54 = 'http://example.com/repo.zip'
    var_55 = True
    var_56 = False
    var_57 = module_1.str(var_50)
    var_58 = 'protected.zip'
    var_59 = var_52 / var_58
    var_60 = 'project/'
    var_61 = 'project/file.txt'
    var_62 = 'password required'
    var_63 = module_1.str(var_59)
    var_64 = False
    var_65 = 'new_dir'
    var_66 = var_63 / var_65
    var_67 = 'test.zip'
    var_68 = var_62 / var_67
    var_69 = 'project/'
    var_70 = ''
    var_71 = 'project/file.txt'
    var_72 = 'content'
    var_73 = module_1.str(var_68)
    var_74 = False
    var_75 = module_2.unzip(var_73, var_74, var_66)
    var_76 = module_0.exists()



