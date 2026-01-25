####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'remote.zip'
    var_19 = 'remote_project/'
    var_20 = ''
    var_21 = 'remote_project/test.txt'
    var_22 = 'data'
    var_23 = 'rb'
    var_24 = 'requests.get'
    var_25 = 'http://example.com/remote.zip'
    var_26 = True
    var_27 = 'pwd.zip'
    var_28 = 'secret'
    var_29 = 'secure_project/'
    var_30 = ''
    var_31 = 'secure_project/file.txt'
    var_32 = 'secret content'
    var_33 = 'utf-8'
    var_34 = module_0.encode(var_33)
    var_35 = False
    var_36 = True
    var_37 = 'cached.zip'
    var_38 = 'cached_project/'
    var_39 = ''
    var_40 = 'cached_project/file.txt'
    var_41 = 'cached'
    var_42 = 'cookiecutter.ziputils.prompt_and_delete'
    var_43 = module_1.patch(var_42)
    var_44 = 'http://example.com/cached.zip'



# Parsed testcases at query #2
#--------------------------


import locale as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = b'PK\x03\x04'
    var_18 = [var_17]
    var_19 = b''
    var_20 = [var_19]
    var_21 = 100
    var_22 = var_20 * var_21
    var_23 = var_18 + var_22
    var_24 = 'from_url.zip'
    var_25 = 'remote_project/'
    var_26 = ''
    var_27 = 'remote_project/file.txt'
    var_28 = 'content'
    var_29 = 'remote_project/'
    var_30 = 'remote_project/file.txt'
    var_31 = False
    var_32 = 'http://example.com/repo.zip'
    var_33 = True
    var_34 = 'protected.zip'
    var_35 = 'secure_project/'
    var_36 = ''
    var_37 = 'secure_project/file.txt'
    var_38 = 'secret'
    var_39 = b'test_password'
    var_40 = 'test_password'
    var_41 = 'project/'
    var_42 = 'project/file.txt'
    var_43 = 'Bad password'
    var_44 = False
    var_45 = 'dummy.zip'
    var_46 = module_0.str(var_42)
    var_47 = False
    var_48 = True
    var_49 = 'project/'
    var_50 = 'project/file.txt'
    var_51 = 'Bad password'
    var_52 = False
    var_53 = 'dummy.zip'
    var_54 = module_0.str(var_50)
    var_55 = False
    var_56 = 'wrong'



# Parsed testcases at query #3
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'test_project'
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'no_dir.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'url_test.zip'
    var_19 = 'project/'
    var_20 = ''
    var_21 = 'project/test.txt'
    var_22 = 'data'
    var_23 = 'clone'
    var_24 = 'http://example.com/test.zip'
    var_25 = True
    var_26 = 'project'
    var_27 = 'protected.zip'
    var_28 = b'secret'
    var_29 = 'secure_project/'
    var_30 = ''
    var_31 = 'secure_project/file.txt'
    var_32 = 'secret content'
    var_33 = 'secure_project/'
    var_34 = ''
    var_35 = 'secure_project/file.txt'
    var_36 = 'secret content'
    var_37 = 'secret'
    var_38 = 'secure_project'
    var_39 = False
    var_40 = True
    var_41 = False
    var_42 = 'wrong'
    var_43 = False
    var_44 = 'new_clone_dir'
    var_45 = module_0.exists()
    var_46 = module_0.exists()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'url_test.zip'
    var_18 = 'remote_project/'
    var_19 = ''
    var_20 = 'remote_project/file.txt'
    var_21 = 'remote content'
    var_22 = 'cloned'
    var_23 = [var_18]
    var_24 = 'http://example.com/test.zip'
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = b'secret'
    var_28 = 'protected_project/'
    var_29 = ''
    var_30 = 'protected_project/file.txt'
    var_31 = 'protected'
    var_32 = 'secure_project/'
    var_33 = ''
    var_34 = 'secure_project/file.txt'
    var_35 = 'content'
    var_36 = False
    var_37 = 'secret'
    var_38 = False
    var_39 = True
    var_40 = False
    var_41 = 'wrong'
    var_42 = False
    var_43 = 'cached.zip'
    var_44 = 'cached_project/'
    var_45 = ''
    var_46 = False



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'valid.zip'
    var_19 = 'my_project/'
    var_20 = ''
    var_21 = 'my_project/README.md'
    var_22 = 'test'
    var_23 = [var_20]
    var_24 = 'https://example.com/repo.zip'
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = b'secret'
    var_28 = 'secure_project/'
    var_29 = ''
    var_30 = 'secure_project/file.txt'
    var_31 = 'secret content'
    var_32 = 'secure_project/'
    var_33 = ''
    var_34 = 'secure_project/file.txt'
    var_35 = 'secret content'
    var_36 = 'secret'
    var_37 = False
    var_38 = True
    var_39 = 'Bad password'
    var_40 = False
    var_41 = 'wrongpwd'



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'bad.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'remote.zip'
    var_19 = 'remote_project/'
    var_20 = ''
    var_21 = 'remote_project/file.txt'
    var_22 = 'content'
    var_23 = 'rb'
    var_24 = [var_2]
    var_25 = 'https://example.com/remote.zip'
    var_26 = True
    var_27 = False
    var_28 = 'protected.zip'
    var_29 = b'test_password'
    var_30 = 'protected_project/'
    var_31 = ''
    var_32 = 'protected_project/file.txt'
    var_33 = 'content'
    var_34 = 'test_password'
    var_35 = False
    var_36 = 'wrong_password'
    var_37 = False
    var_38 = True
    var_39 = 'https://example.com/remote.zip'
    var_40 = True



# Parsed testcases at query #8
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'zips'
    var_2 = 'extract'
    var_3 = 'test.zip'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'notoplevel.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'bad.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = b'test_content'
    var_19 = [var_18]
    var_20 = 'project/'
    var_21 = 'project/file.txt'
    var_22 = 'https://example.com/project.zip'
    var_23 = True
    var_24 = module_0.unzip(var_22, var_23, var_7)
    var_25 = 'project/'
    var_26 = 'project/file.txt'
    var_27 = 'Bad password'
    var_28 = None
    var_29 = False
    var_30 = 'correct_password'
    var_31 = module_0.unzip(var_3, var_29, var_8, password=var_30)
    var_32 = 'project/'
    var_33 = 'project/file.txt'
    var_34 = 'Bad password'
    var_35 = False
    var_36 = 'wrong_password'
    var_37 = module_0.unzip(var_32, var_35, var_34, password=var_36)
    var_38 = 'project/'
    var_39 = 'project/file.txt'
    var_40 = 'Bad password'
    var_41 = False
    var_42 = True
    var_43 = module_0.unzip(var_38, var_41, var_40, var_42)
    var_44 = 'project/'
    var_45 = 'project/file.txt'
    var_46 = 'cache'
    var_47 = 'test.zip'
    var_48 = 'cached'
    var_49 = 'https://example.com/test.zip'
    var_50 = True



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = b'PK\x03\x04'
    var_18 = [var_17]
    var_19 = 'project/'
    var_20 = 'project/file.txt'
    var_21 = 'http://example.com/project.zip'
    var_22 = True
    var_23 = 'project/'
    var_24 = 'project/file.txt'
    var_25 = 'Bad password'
    var_26 = None
    var_27 = False
    var_28 = 'correct_password'
    var_29 = 'project/'
    var_30 = 'project/file.txt'
    var_31 = 'Bad password'
    var_32 = False
    var_33 = 'wrong_password'
    var_34 = 'project/'
    var_35 = 'project/file.txt'
    var_36 = 'Bad password'
    var_37 = False
    var_38 = True



# Parsed testcases at query #10
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'https://example.com/project.zip'
    var_18 = b'test'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'project_name/'
    var_22 = ''
    var_23 = 'project_name/file.txt'
    var_24 = 'content'
    var_25 = 'builtins.open'
    var_26 = 'expanduser'
    var_27 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_28 = module_0.patch(var_27)
    var_29 = 'cookiecutter.zipfile.prompt_and_delete'
    var_30 = True
    var_31 = module_0.patch(var_29)
    var_32 = 'zipfile.ZipFile'
    var_33 = module_0.patch(var_32)
    var_34 = 'protected.zip'
    var_35 = b'password'
    var_36 = 'project_name/'
    var_37 = ''
    var_38 = 'project_name/file.txt'
    var_39 = 'content'
    var_40 = 'cookiecutter.zipfile.read_repo_password'
    var_41 = 'password'
    var_42 = module_0.patch(var_40)
    var_43 = 'project_name/'
    var_44 = 'Bad password'
    var_45 = False
    var_46 = 'wrong'
    var_47 = False
    var_48 = True



# Parsed testcases at query #11
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'remote.zip'
    var_19 = 'remote_project/'
    var_20 = ''
    var_21 = 'remote_project/file.txt'
    var_22 = 'remote content'
    var_23 = [var_19]
    var_24 = 'https://example.com/remote.zip'
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = 'test123'
    var_28 = 'utf-8'
    var_29 = module_0.encode(var_28)
    var_30 = 'protected_project/'
    var_31 = ''
    var_32 = 'protected_project/file.txt'
    var_33 = 'protected content'
    var_34 = 'protected_project/'
    var_35 = ''
    var_36 = 'protected_project/file.txt'
    var_37 = 'protected content'
    var_38 = 'cached.zip'
    var_39 = 'cached_project/'
    var_40 = ''
    var_41 = 'cached_project/file.txt'
    var_42 = 'cached'
    var_43 = 'https://example.com/cached.zip'
    var_44 = True
    var_45 = False



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip with an empty zip file.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip with zip file missing top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip with an invalid zip file.'
    var_1 = 'bad.zip'
    var_2 = 'extract'
    var_3 = 'This is not a zip file'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip with URL download.'
    var_1 = 'clone'
    var_2 = 'content.zip'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'http://example.com/test.zip'
    var_8 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project/'
    var_5 = ''
    var_6 = 'project/file.txt'
    var_7 = 'content'
    var_8 = 'utf-8'
    var_9 = module_0.encode(var_8)
    var_10 = False

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'correct_password'
    var_4 = 'project/'
    var_5 = ''
    var_6 = 'project/file.txt'
    var_7 = 'content'
    var_8 = 'utf-8'
    var_9 = module_0.encode(var_8)
    var_10 = False
    var_11 = 'wrong_password'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project/'
    var_5 = ''
    var_6 = 'project/file.txt'
    var_7 = 'content'
    var_8 = 'utf-8'
    var_9 = module_0.encode(var_8)
    var_10 = False
    var_11 = True

def test_case_0():
    var_0 = 'Test unzip with cached zip file and no_input=True.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True



# Parsed testcases at query #13
#--------------------------


import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'https://example.com/archive.zip'
    var_9 = 'clone'
    var_10 = 'archive.zip'
    var_11 = 'test_project/'
    var_12 = ''
    var_13 = 'test_project/README.md'
    var_14 = 'test'
    var_15 = b'test_data'
    var_16 = [var_15]
    var_17 = 'requests.get'
    var_18 = 'builtins.open'
    var_19 = True
    var_20 = 'empty.zip'
    var_21 = False
    var_22 = 'bad.zip'
    var_23 = 'file.txt'
    var_24 = 'content'
    var_25 = False
    var_26 = 'invalid.zip'
    var_27 = 'not a zip file'
    var_28 = False
    var_29 = 'protected.zip'
    var_30 = 'testpass'
    var_31 = 'secure_project/'
    var_32 = ''
    var_33 = 'secure_project/secret.txt'
    var_34 = 'secret'
    var_35 = 'utf-8'
    var_36 = module_0.encode(var_35)
    var_37 = False
    var_38 = 'wrongpass'
    var_39 = False
    var_40 = True
    var_41 = 'cookiecutter.prompt.prompt_and_delete'
    var_42 = module_1.patch(var_41)



# Parsed testcases at query #14
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'test'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'repo.zip'
    var_22 = 'project/'
    var_23 = ''
    var_24 = 'project/file.txt'
    var_25 = 'content'
    var_26 = 'cookiecutter.ziputil.prompt_and_delete'
    var_27 = True
    var_28 = module_0.patch(var_26)
    var_29 = 'http://example.com/repo.zip'
    var_30 = 'protected.zip'
    var_31 = b'password'
    var_32 = 'secure/'
    var_33 = ''
    var_34 = 'secure/file.txt'
    var_35 = 'secret'
    var_36 = 'password'
    var_37 = module_0.patch(var_26)
    var_38 = False
    var_39 = True
    var_40 = False
    var_41 = 'wrongpassword'



# Parsed testcases at query #15
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'notopdir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'PK\x03\x04'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'builtins.open'
    var_22 = 'url_test.zip'
    var_23 = 'remote_project/'
    var_24 = ''
    var_25 = 'remote_project/file.txt'
    var_26 = 'content'
    var_27 = 'os.path.exists'
    var_28 = module_0.patch(var_27)
    var_29 = 'zipfile.ZipFile'
    var_30 = module_0.patch(var_29)
    var_31 = 'remote_project/'
    var_32 = 'remote_project/file.txt'
    var_33 = 'http://example.com/archive.zip'
    var_34 = True
    var_35 = 'protected.zip'
    var_36 = b'secret'
    var_37 = 'protected_project/'
    var_38 = ''
    var_39 = 'secret'
    var_40 = False
    var_41 = 'wrong'
    var_42 = False
    var_43 = True
    var_44 = 'cookiecutter.prompt.read_repo_password'
    var_45 = 'wrong'
    var_46 = [var_45, var_45, var_45]
    var_47 = module_0.patch(var_44)
    var_48 = False



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'bad.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'test'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'repo.zip'
    var_22 = 'repo/'
    var_23 = ''
    var_24 = 'repo/file.txt'
    var_25 = 'content'
    var_26 = 'http://example.com/repo.zip'
    var_27 = True
    var_28 = 'protected.zip'
    var_29 = b'secret'
    var_30 = 'secure_project/'
    var_31 = ''
    var_32 = 'secure_project/file.txt'
    var_33 = 'content'
    var_34 = 'secret'
    var_35 = False
    var_36 = 'wrong'
    var_37 = False
    var_38 = True
    var_39 = 'cookiecutter.prompt.read_repo_password'
    var_40 = 'wrong1'
    var_41 = 'wrong2'
    var_42 = 'wrong3'
    var_43 = [var_40, var_41, var_42]
    var_44 = module_0.patch(var_39)
    var_45 = False
    var_46 = 'cookiecutter.prompt.prompt_and_delete'
    var_47 = module_0.patch(var_46)



# Parsed testcases at query #2
#--------------------------


import requests.api as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'no_dir.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_19 = module_0.patch(var_18)
    var_20 = b'test'
    var_21 = [var_20]
    var_22 = 'cookiecutter.ziputils.requests.get'
    var_23 = 'cookiecutter.ziputils.prompt_and_delete'
    var_24 = module_0.patch(var_23)
    var_25 = 'builtins.open'
    var_26 = 'valid_url.zip'
    var_27 = 'project/'
    var_28 = ''
    var_29 = 'project/file.txt'
    var_30 = 'content'
    var_31 = 'https://example.com/project.zip'
    var_32 = 'protected.zip'
    var_33 = 'test_password'
    var_34 = 'utf-8'
    var_35 = module_1.encode(var_34)
    var_36 = 'secure_project/'
    var_37 = ''
    var_38 = 'secure_project/file.txt'
    var_39 = 'secret'
    var_40 = False
    var_41 = True
    var_42 = False
    var_43 = True
    var_44 = 'wrong_password'



# Parsed testcases at query #3
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test_zip'
    var_2 = 'test.zip'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'http://example.com/test.zip'
    var_18 = b'test'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'downloaded.zip'
    var_22 = 'project/'
    var_23 = ''
    var_24 = 'project/file.txt'
    var_25 = 'content'
    var_26 = 'builtins.open'
    var_27 = 'protected.zip'
    var_28 = 'secret'
    var_29 = 'utf-8'
    var_30 = module_0.encode(var_29)
    var_31 = 'secure_project/'
    var_32 = ''
    var_33 = 'secure_project/file.txt'
    var_34 = 'secret content'
    var_35 = False
    var_36 = 'wrong'
    var_37 = False
    var_38 = True
    var_39 = 'clone_test'
    var_40 = 'valid.zip'
    var_41 = 'myproject/'
    var_42 = ''
    var_43 = 'myproject/README.md'
    var_44 = 'readme'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'zips'
    var_2 = 'extract'
    var_3 = 'test.zip'
    var_4 = 'test_project/'
    var_5 = ''
    var_6 = 'test_project/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'test_project'
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'bad.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
    var_15 = False
    var_16 = 'notazip.zip'
    var_17 = 'not a zip file'
    var_18 = False
    var_19 = b'test'
    var_20 = 'url_test.zip'
    var_21 = 'url_project/'
    var_22 = ''
    var_23 = 'url_project/file.txt'
    var_24 = 'content'
    var_25 = 'url_project/'
    var_26 = 'url_project/file.txt'
    var_27 = 'http://example.com/test.zip'
    var_28 = True
    var_29 = 'protected_project/'
    var_30 = 'protected_project/file.txt'
    var_31 = 'Bad password'
    var_32 = None
    var_33 = False
    var_34 = 'correct_password'
    var_35 = 'project/'
    var_36 = 'project/file.txt'
    var_37 = 'Bad password'
    var_38 = False
    var_39 = True
    var_40 = 'project/'
    var_41 = 'project/file.txt'
    var_42 = 'Bad password'
    var_43 = None
    var_44 = 'wrong1'
    var_45 = 'wrong2'
    var_46 = 'correct'
    var_47 = False
    var_48 = 'project/'
    var_49 = 'project/file.txt'
    var_50 = 'Bad password'
    var_51 = False



# Parsed testcases at query #5
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = b'test content'
    var_2 = [var_1]
    var_3 = 'requests.get'
    var_4 = 'test.zip'
    var_5 = 'project_name/'
    var_6 = ''
    var_7 = 'project_name/file.txt'
    var_8 = 'content'
    var_9 = 'builtins.open'
    var_10 = 'os.path.exists'
    var_11 = False
    var_12 = module_0.patch(var_10)
    var_13 = 'cookiecutter.utils.make_sure_path_exists'
    var_14 = module_0.patch(var_13)
    var_15 = 'tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = 'extractall'
    var_18 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.utils.make_sure_path_exists'
    var_7 = module_0.patch(var_6)
    var_8 = 'tempfile.mkdtemp'
    var_9 = 'temp'
    var_10 = 'extractall'
    var_11 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with empty zip file raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.utils.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'tempfile.mkdtemp'
    var_5 = 'temp'
    var_6 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.utils.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = 'tempfile.mkdtemp'
    var_7 = 'temp'
    var_8 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = 'not a zip file'
    var_3 = 'cookiecutter.utils.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False

import requests.api as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.utils.make_sure_path_exists'
    var_7 = module_0.patch(var_6)
    var_8 = 'tempfile.mkdtemp'
    var_9 = 'temp'
    var_10 = 'extractall'
    var_11 = module_1.RuntimeError()
    var_12 = None
    var_13 = False
    var_14 = 'test_password'

import requests.api as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'Test unzip with invalid password raises InvalidZipRepository.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'cookiecutter.utils.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = 'tempfile.mkdtemp'
    var_7 = 'temp'
    var_8 = 'extractall'
    var_9 = module_1.RuntimeError()
    var_10 = False
    var_11 = 'wrong_password'

import requests.api as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input raises InvalidZipRepository.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'cookiecutter.utils.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = 'tempfile.mkdtemp'
    var_7 = 'temp'
    var_8 = 'extractall'
    var_9 = module_1.RuntimeError()
    var_10 = False
    var_11 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and user chooses to delete.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.utils.make_sure_path_exists'
    var_7 = module_0.patch(var_6)
    var_8 = 'os.path.exists'
    var_9 = True
    var_10 = module_0.patch(var_8)
    var_11 = 'cookiecutter.prompt.prompt_and_delete'
    var_12 = False
    var_13 = module_0.patch(var_11)
    var_14 = 'tempfile.mkdtemp'
    var_15 = 'temp'
    var_16 = 'extractall'
    var_17 = 'http://example.com/test.zip'



# Parsed testcases at query #6
#--------------------------


import requests.api as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test_zip'
    var_2 = 'valid.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project_name'
    var_9 = 'file.txt'
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = 'invalid.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
    var_15 = False
    var_16 = 'bad.zip'
    var_17 = b'not a zip file'
    var_18 = False
    var_19 = 'url_test.zip'
    var_20 = 'remote_project/'
    var_21 = ''
    var_22 = 'remote_project/file.txt'
    var_23 = 'content'
    var_24 = 'rb'
    var_25 = 'requests.get'
    var_26 = 'cookiecutter.prompt.prompt_and_delete'
    var_27 = True
    var_28 = module_0.patch(var_26)
    var_29 = 'http://example.com/remote_project.zip'
    var_30 = 'remote_project'
    var_31 = 'protected.zip'
    var_32 = 'test_password'
    var_33 = 'secure_project/'
    var_34 = ''
    var_35 = 'secure_project/file.txt'
    var_36 = 'content'
    var_37 = 'protected_new.zip'
    var_38 = 'utf-8'
    var_39 = module_1.encode(var_38)
    var_40 = 'secure_project/'
    var_41 = ''
    var_42 = 'secure_project/file.txt'
    var_43 = 'content'
    var_44 = 'secure_project'
    var_45 = b'password'
    var_46 = 'pwd_project/'
    var_47 = ''
    var_48 = 'zipfile.ZipFile.extractall'
    var_49 = 'Bad password'
    var_50 = False
    var_51 = True
    var_52 = '~/test_cookiecutter'
    var_53 = 'pathlib.Path.expanduser'
    var_54 = 'cookiecutter.utils.make_sure_path_exists'
    var_55 = module_0.patch(var_54)
    var_56 = 'home_test.zip'
    var_57 = 'home_project/'
    var_58 = ''
    var_59 = 'home_project'



# Parsed testcases at query #7
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'test_project'
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'no_toplevel.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'bad.zip'
    var_16 = b'not a zip file'
    var_17 = False
    var_18 = 'clone'
    var_19 = 'url_test.zip'
    var_20 = 'url_project/'
    var_21 = ''
    var_22 = 'url_project/file.txt'
    var_23 = 'content'
    var_24 = 'requests.get'
    var_25 = 'cookiecutter.utils.make_sure_path_exists'
    var_26 = module_0.patch(var_25)
    var_27 = 'http://example.com/url_test.zip'
    var_28 = True
    var_29 = 'protected.zip'
    var_30 = b'password'
    var_31 = 'protected_project/'
    var_32 = ''
    var_33 = 'protected_project/file.txt'
    var_34 = 'secret'
    var_35 = 'password'
    var_36 = 'protected_project'
    var_37 = False
    var_38 = 'wrongpassword'
    var_39 = False
    var_40 = True
    var_41 = 'home_test.zip'
    var_42 = 'home_project/'
    var_43 = ''
    var_44 = 'pathlib.Path.expanduser'
    var_45 = '~'



# Parsed testcases at query #8
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'https://example.com/repo.zip'
    var_18 = b'PK\x03\x04'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'cookiecutter.ziputil.ZipFile'
    var_22 = module_0.patch(var_21)
    var_23 = 'os.path.exists'
    var_24 = module_0.patch(var_23)
    var_25 = 'cookiecutter.ziputil.prompt_and_delete'
    var_26 = True
    var_27 = module_0.patch(var_25)
    var_28 = 'temp.zip'
    var_29 = 'myproject/'
    var_30 = ''
    var_31 = 'myproject/file.txt'
    var_32 = 'content'
    var_33 = 'myproject/'
    var_34 = 'myproject/file.txt'
    var_35 = None
    var_36 = 'clone'
    var_37 = 'protected.zip'
    var_38 = b'secret'
    var_39 = 'secure/'
    var_40 = ''
    var_41 = 'secure/file.txt'
    var_42 = 'content'
    var_43 = 'secret'
    var_44 = False
    var_45 = True



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'project-name/'
    var_3 = ''
    var_4 = 'project-name/file.txt'
    var_5 = 'content'
    var_6 = False

def test_case_0():
    var_0 = 'Test unzip function with URL download.'
    var_1 = 'test.zip'
    var_2 = 'project-name/'
    var_3 = ''
    var_4 = 'project-name/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True

def test_case_0():
    var_0 = 'Test unzip with empty zip file.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip with zip file missing top-level directory.'
    var_1 = 'bad.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip with invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected.zip'
    var_2 = 'secret'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'project-name/'
    var_3 = ''
    var_4 = 'project-name/file.txt'
    var_5 = 'content'
    var_6 = 'project-name/'
    var_7 = 'project-name/file.txt'
    var_8 = 'Bad password'
    var_9 = False
    var_10 = False
    var_11 = True

def test_case_0():
    var_0 = 'Test unzip with existing cached file.'
    var_1 = 'test.zip'
    var_2 = 'project-name/'
    var_3 = ''
    var_4 = 'project-name/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = False



# Parsed testcases at query #10
#--------------------------


import requests.api as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'nodir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'test'
    var_19 = 'requests.get'
    var_20 = 'downloaded.zip'
    var_21 = 'test_project/'
    var_22 = ''
    var_23 = 'test_project/file.txt'
    var_24 = 'content'
    var_25 = 'builtins.open'
    var_26 = 'os.path.exists'
    var_27 = module_0.patch(var_26)
    var_28 = 'http://example.com/test.zip'
    var_29 = True
    var_30 = 'protected.zip'
    var_31 = 'test_password'
    var_32 = 'utf-8'
    var_33 = module_1.encode(var_32)
    var_34 = 'secure_project/'
    var_35 = ''
    var_36 = 'secure_project/file.txt'
    var_37 = 'secret'
    var_38 = False
    var_39 = 'wrong_password'
    var_40 = False
    var_41 = True
    var_42 = 'cookiecutter.prompt.read_repo_password'
    var_43 = 'wrong1'
    var_44 = 'wrong2'
    var_45 = 'wrong3'
    var_46 = [var_43, var_44, var_45]
    var_47 = module_0.patch(var_42)
    var_48 = False
    var_49 = 'cached'
    var_50 = 'existing.zip'
    var_51 = 'existing/'
    var_52 = ''
    var_53 = 'cookiecutter.prompt.prompt_and_delete'
    var_54 = module_0.patch(var_53)
    var_55 = module_0.patch(var_19)
    var_56 = 'http://example.com/existing.zip'



# Parsed testcases at query #11
#--------------------------


import requests.api as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'content'
    var_2 = 'project-name'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'test.zip'
    var_6 = 'project-name'
    var_7 = 'project-name/'
    var_8 = 'file.txt'
    var_9 = 'project-name/file.txt'
    var_10 = 'clone'
    var_11 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_12 = module_0.patch(var_11)
    var_13 = 'cookiecutter.ziputils.prompt_and_delete'
    var_14 = False
    var_15 = module_0.patch(var_13)
    var_16 = 'cookiecutter.ziputils.requests.get'
    var_17 = module_0.patch(var_16)
    var_18 = module_1.exists()

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with empty zip file.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with zip file missing top-level directory.'
    var_1 = 'no_topdir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = b'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip downloading from URL.'
    var_1 = 'content'
    var_2 = 'project-name'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'test.zip'
    var_6 = 'project-name'
    var_7 = 'project-name/'
    var_8 = 'file.txt'
    var_9 = 'project-name/file.txt'
    var_10 = 'clone'
    var_11 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_12 = module_0.patch(var_11)
    var_13 = 'cookiecutter.ziputils.prompt_and_delete'
    var_14 = True
    var_15 = module_0.patch(var_13)
    var_16 = 'cookiecutter.ziputils.requests.get'
    var_17 = 'http://example.com/test.zip'

import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'utf-8'
    var_6 = module_0.encode(var_5)
    var_7 = 'project-name/file.txt'
    var_8 = 'content'
    var_9 = 'clone'
    var_10 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_11 = module_1.patch(var_10)
    var_12 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with invalid password for protected zip.'
    var_1 = 'protected.zip'
    var_2 = 'project-name/'
    var_3 = ''
    var_4 = b'correctpass'
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.ziputils.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = False
    var_11 = 'wrongpass'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip raises error for empty zip file.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_dir.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = b'invalid zip content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip downloads from URL.'
    var_1 = 'clone'
    var_2 = b'PK\x03\x04'
    var_3 = [var_2]
    var_4 = 'project/'
    var_5 = 'http://example.com/repo.zip'
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'testpass'

def test_case_0():
    var_0 = 'Test unzip raises error for invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/'
    var_6 = 'Bad password'
    var_7 = False
    var_8 = 'wrong'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip expands user path.'
    var_1 = 'project/'
    var_2 = '~/test.zip'
    var_3 = False
    var_4 = '~/cookiecutter'
    var_5 = module_0.unzip(var_2, var_3, var_4)



# Parsed testcases at query #13
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = '/nonexistent/path/repo.zip'
    var_2 = False
    var_3 = '.'
    var_4 = True
    var_5 = module_0.unzip(var_1, var_2, var_3, var_4)
    var_6 = 'test.zip'
    var_7 = 'test_project/'
    var_8 = ''
    var_9 = 'test_project/file.txt'
    var_10 = 'content'
    var_11 = False
    var_12 = True
    var_13 = 'empty.zip'
    var_14 = False
    var_15 = True
    var_16 = 'no_topdir.zip'
    var_17 = 'file.txt'
    var_18 = 'content'
    var_19 = False
    var_20 = True
    var_21 = 'repo.zip'
    var_22 = 'myproject/'
    var_23 = ''
    var_24 = 'myproject/test.txt'
    var_25 = 'data'
    var_26 = 'http://example.com/repo.zip'
    var_27 = True
    var_28 = 'project/'
    var_29 = ''
    var_30 = 'project/file.txt'
    var_31 = 'content'
    var_32 = b'test'
    var_33 = [var_32]
    var_34 = 'project/'
    var_35 = 'project/file.txt'
    var_36 = 'http://example.com/repo.zip'
    var_37 = True
    var_38 = '/path/to/protected.zip'
    var_39 = False
    var_40 = True
    var_41 = 'mypassword'
    var_42 = 'invalid.zip'
    var_43 = 'not a zip file'
    var_44 = False
    var_45 = True



# Parsed testcases at query #14
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'test content'
    var_19 = [var_18]
    var_20 = 'requests.get'
    var_21 = 'cookiecutter.repository.unzip.prompt_and_delete'
    var_22 = True
    var_23 = module_0.patch(var_21)
    var_24 = 'url_test.zip'
    var_25 = 'my_project/'
    var_26 = ''
    var_27 = 'my_project/file.txt'
    var_28 = 'content'
    var_29 = 'https://example.com/my_project.zip'
    var_30 = 'protected.zip'
    var_31 = b'secret'
    var_32 = 'secure_project/'
    var_33 = ''
    var_34 = 'secure_project/file.txt'
    var_35 = 'content'
    var_36 = 'secret'
    var_37 = False
    var_38 = True
    var_39 = False
    var_40 = 'wrong'
    var_41 = 'cookiecutter.repository.unzip.read_repo_password'
    var_42 = 'wrong'
    var_43 = module_0.patch(var_41)
    var_44 = False
    var_45 = 'clone2'
    var_46 = 'cached.zip'
    var_47 = 'cached_project/'
    var_48 = ''
    var_49 = 'cached_project/file.txt'
    var_50 = 'content'
    var_51 = module_0.patch(var_21)



# Parsed testcases at query #15
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'bad.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'url_test.zip'
    var_18 = 'remote_project/'
    var_19 = ''
    var_20 = 'remote_project/file.txt'
    var_21 = 'remote content'
    var_22 = 'clone'
    var_23 = 'http://example.com/remote_project.zip'
    var_24 = True
    var_25 = 'protected.zip'
    var_26 = b'test_password'
    var_27 = 'secure_project/'
    var_28 = ''
    var_29 = 'secure_project/secret.txt'
    var_30 = 'secret'
    var_31 = 'test_password'
    var_32 = False
    var_33 = True
    var_34 = False
    var_35 = 'wrong_password'
    var_36 = 'new_clone'
    var_37 = 'nested'
    var_38 = 'normal.zip'
    var_39 = 'project/'
    var_40 = ''
    var_41 = module_0.exists()



# Parsed testcases at query #16
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'clone'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.repository.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = b'test'
    var_10 = 'cookiecutter.repository.prompt_and_delete'
    var_11 = True
    var_12 = module_0.patch(var_10)
    var_13 = 'os.path.exists'
    var_14 = False
    var_15 = module_0.patch(var_13)
    var_16 = 'builtins.open'
    var_17 = 'http://example.com/test.zip'
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'bad.zip'
    var_21 = 'file.txt'
    var_22 = 'content'
    var_23 = False
    var_24 = 'invalid.zip'
    var_25 = 'not a zip file'
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = 'project_name/'
    var_29 = ''
    var_30 = 'project_name/file.txt'
    var_31 = 'content'
    var_32 = 'test_password'
    var_33 = 'extractall'
    var_34 = 'Bad password'
    var_35 = False
    var_36 = True
    var_37 = module_0.patch(var_13)
    var_38 = module_0.patch(var_10)
    var_39 = 'http://example.com/cached.zip'



# Parsed testcases at query #17
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'valid.zip'
    var_18 = 'myproject/'
    var_19 = ''
    var_20 = 'myproject/README.md'
    var_21 = 'readme content'
    var_22 = 'https://example.com/test.zip'
    var_23 = True
    var_24 = 'protected.zip'
    var_25 = 'secret123'
    var_26 = 'utf-8'
    var_27 = module_0.encode(var_26)
    var_28 = 'protected_project/'
    var_29 = ''
    var_30 = 'protected_project/file.txt'
    var_31 = 'content'
    var_32 = False
    var_33 = 'wrongpassword'
    var_34 = False
    var_35 = True
    var_36 = 'Bad password'
    var_37 = None



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'my_project'
    var_4 = 'my_project/'
    var_5 = ''
    var_6 = 'my_project/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'bad.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'url_test.zip'
    var_19 = 'project/'
    var_20 = ''
    var_21 = 'project/file.txt'
    var_22 = 'content'
    var_23 = [var_19]
    var_24 = 'http://example.com/project.zip'
    var_25 = True
    var_26 = 'pwd.zip'
    var_27 = b'test123'
    var_28 = 'secure_project/'
    var_29 = ''
    var_30 = 'secure_project/file.txt'
    var_31 = 'secret'
    var_32 = 'password required'
    var_33 = None
    var_34 = False
    var_35 = 'test123'
    var_36 = False
    var_37 = True
    var_38 = 'password required'
    var_39 = None
    var_40 = False
    var_41 = False
    var_42 = 'wrong'
    var_43 = False
    var_44 = 'cached.zip'
    var_45 = 'cached_project/'
    var_46 = ''
    var_47 = 'http://example.com/cached.zip'
    var_48 = True



# Parsed testcases at query #19
#--------------------------


import builtins as module_0

def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_topdir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = b'test'
    var_18 = [var_17]
    var_19 = 'project/'
    var_20 = 'project/file.txt'
    var_21 = None
    var_22 = 'http://example.com/repo.zip'
    var_23 = True
    var_24 = 'project/'
    var_25 = 'project/file.txt'
    var_26 = module_0.RuntimeError()
    var_27 = None
    var_28 = [var_26, var_27]
    var_29 = False
    var_30 = 'test_password'
    var_31 = 'project/'
    var_32 = 'project/file.txt'
    var_33 = module_0.RuntimeError()
    var_34 = None
    var_35 = False
    var_36 = 'wrong_password'
    var_37 = 'project/'
    var_38 = 'project/file.txt'
    var_39 = module_0.RuntimeError()
    var_40 = None
    var_41 = False
    var_42 = True



# Parsed testcases at query #20
#--------------------------


import email._encoded_words as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_top_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = b'PK\x03\x04'
    var_18 = [var_17]
    var_19 = 'url_test.zip'
    var_20 = 'project/'
    var_21 = ''
    var_22 = 'project/test.txt'
    var_23 = 'data'
    var_24 = 'http://example.com/project.zip'
    var_25 = True
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = 'test_password'
    var_29 = 'project/'
    var_30 = ''
    var_31 = 'project/file.txt'
    var_32 = 'secret'
    var_33 = 'utf-8'
    var_34 = module_0.encode(var_33)
    var_35 = 'project/'
    var_36 = ''
    var_37 = 'project/file.txt'
    var_38 = 'secret'
    var_39 = False
    var_40 = True
    var_41 = 'new_clone_dir'
    var_42 = module_1.exists()
    var_43 = 'valid.zip'
    var_44 = 'project/'
    var_45 = ''
    var_46 = 'project/file.txt'
    var_47 = 'content'
    var_48 = module_1.exists()



# Parsed testcases at query #21
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'bad.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = b'test content'
    var_18 = 'url_test.zip'
    var_19 = 'project/'
    var_20 = ''
    var_21 = 'project/file.txt'
    var_22 = 'content'
    var_23 = 'project/'
    var_24 = 'project/file.txt'
    var_25 = 'http://example.com/test.zip'
    var_26 = True
    var_27 = 'protected.zip'
    var_28 = b'secret'
    var_29 = 'secure_project/'
    var_30 = ''
    var_31 = 'secure_project/file.txt'
    var_32 = 'secret content'
    var_33 = 'secret'
    var_34 = False
    var_35 = True
    var_36 = False
    var_37 = 'wrongpassword'
    var_38 = 'new_clone_dir'
    var_39 = module_0.exists()
    var_40 = 'project/'
    var_41 = ''
    var_42 = module_0.exists()



# Parsed testcases at query #22
#--------------------------


import requests.api as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'notoplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = b'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'remote.zip'
    var_19 = 'remote_project/'
    var_20 = ''
    var_21 = 'remote_project/file.txt'
    var_22 = 'content'
    var_23 = 'requests.get'
    var_24 = 'cookiecutter.prompt.prompt_and_delete'
    var_25 = True
    var_26 = module_0.patch(var_24)
    var_27 = 'http://example.com/remote.zip'
    var_28 = 'protected.zip'
    var_29 = 'secret'
    var_30 = 'utf-8'
    var_31 = module_1.encode(var_30)
    var_32 = 'protected_project/'
    var_33 = ''
    var_34 = 'protected_project/file.txt'
    var_35 = 'content'
    var_36 = False
    var_37 = 'wrongpassword'
    var_38 = False
    var_39 = True
    var_40 = 'cookiecutter.prompt.read_repo_password'
    var_41 = module_0.patch(var_40)



# Parsed testcases at query #23
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'project'
    var_3 = 'test-project/'
    var_4 = ''
    var_5 = 'test-project/file.txt'
    var_6 = 'content'
    var_7 = 'https://example.com/test.zip'
    var_8 = True
    var_9 = 'local.zip'
    var_10 = 'local-project'
    var_11 = 'local-project/'
    var_12 = ''
    var_13 = 'local-project/readme.txt'
    var_14 = 'readme content'
    var_15 = False
    var_16 = 'empty.zip'
    var_17 = False
    var_18 = 'bad.zip'
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = False
    var_22 = 'invalid.zip'
    var_23 = 'not a zip file'
    var_24 = False
    var_25 = 'protected.zip'
    var_26 = 'testpass'
    var_27 = 'protected-project/'
    var_28 = ''
    var_29 = 'protected-project/file.txt'
    var_30 = 'content'
    var_31 = 'utf-8'
    var_32 = module_0.encode(var_31)
    var_33 = 'protected-project/'
    var_34 = ''
    var_35 = 'protected-project/file.txt'
    var_36 = 'content'
    var_37 = '~'



# Parsed testcases at query #24
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_top_level.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'url_test.zip'
    var_18 = 'remote_project/'
    var_19 = ''
    var_20 = 'remote_project/file.txt'
    var_21 = 'remote content'
    var_22 = 'clone'
    var_23 = 'MockResponse'
    var_24 = ()
    var_25 = 'iter_content'
    var_26 = 'requests.get'
    var_27 = 'cookiecutter.zipfile_utils.prompt_and_delete'
    var_28 = True
    var_29 = lambda *args, **kwargs: var_28
    var_30 = 'http://example.com/remote_project.zip'
    var_31 = 'password.zip'
    var_32 = 'test123'
    var_33 = 'secure_project/'
    var_34 = ''
    var_35 = 'secure_project/secret.txt'
    var_36 = 'secret content'
    var_37 = 'utf-8'
    var_38 = module_0.encode(var_37)
    var_39 = False
    var_40 = 'wrongpassword'
    var_41 = False
    var_42 = True



# Parsed testcases at query #25
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'valid.zip'
    var_18 = 'myproject/'
    var_19 = ''
    var_20 = 'myproject/README.md'
    var_21 = '# My Project'
    var_22 = 'cache'
    var_23 = [var_18]
    var_24 = 'http://example.com/valid.zip'
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = 'secret123'
    var_28 = 'utf-8'
    var_29 = module_0.encode(var_28)
    var_30 = 'secure_project/'
    var_31 = ''
    var_32 = 'secure_project/data.txt'
    var_33 = 'sensitive'
    var_34 = False
    var_35 = 'wrong'
    var_36 = 'project/'
    var_37 = 'Bad password'
    var_38 = False
    var_39 = True
    var_40 = 'project/'
    var_41 = 'Bad password'
    var_42 = False



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import email._encoded_words as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = True
    var_13 = 'bad.zip'
    var_14 = 'file.txt'
    var_15 = 'content'
    var_16 = False
    var_17 = True
    var_18 = 'invalid.zip'
    var_19 = 'not a zip file'
    var_20 = False
    var_21 = True
    var_22 = 'download.zip'
    var_23 = 'remote_project/'
    var_24 = ''
    var_25 = 'remote_project/file.txt'
    var_26 = 'content'
    var_27 = [var_24]
    var_28 = 'https://example.com/repo.zip'
    var_29 = True
    var_30 = 'remote_project'
    var_31 = 'protected.zip'
    var_32 = 'test_password'
    var_33 = 'utf-8'
    var_34 = module_0.encode(var_33)
    var_35 = 'protected_project/'
    var_36 = ''
    var_37 = 'protected_project/file.txt'
    var_38 = 'content'
    var_39 = 'protected_project'
    var_40 = False
    var_41 = True
    var_42 = 'new_clone'
    var_43 = 'subdir'
    var_44 = module_1.exists()



# Parsed testcases at query #2
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'test-project/'
    var_3 = ''
    var_4 = 'test-project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'file.txt'
    var_8 = module_0.exists(var_5)

def test_case_0():
    var_0 = 'Test unzip raises error for empty zip file.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip downloads from URL correctly.'
    var_1 = False
    var_2 = '.zip'
    var_3 = 'test-project/'
    var_4 = ''
    var_5 = 'test-project/file.txt'
    var_6 = 'content'
    var_7 = 'https://example.com/test.zip'
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and password provided.'
    var_1 = 'protected.zip'
    var_2 = 'test-project/'
    var_3 = ''
    var_4 = 'test-project/file.txt'
    var_5 = 'content'
    var_6 = b'mypassword'
    var_7 = 'test-project/'
    var_8 = 'test-project/file.txt'
    var_9 = 'Bad password'
    var_10 = None
    var_11 = False
    var_12 = 'mypassword'

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zip with no_input=True.'
    var_1 = 'test-project/'
    var_2 = 'test-project/file.txt'
    var_3 = 'Bad password'
    var_4 = 'test.zip'
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip raises error for invalid password.'
    var_1 = 'test-project/'
    var_2 = 'test-project/file.txt'
    var_3 = 'Bad password'



# Parsed testcases at query #3
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'valid.zip'
    var_19 = 'my_project/'
    var_20 = ''
    var_21 = 'my_project/template.txt'
    var_22 = 'template content'
    var_23 = 'requests.get'
    var_24 = 'cookiecutter.utils.make_sure_path_exists'
    var_25 = module_0.patch(var_24)
    var_26 = 'cookiecutter.prompt.prompt_and_delete'
    var_27 = True
    var_28 = module_0.patch(var_26)
    var_29 = 'https://example.com/repo.zip'
    var_30 = 'protected.zip'
    var_31 = b'password'
    var_32 = 'secure_project/'
    var_33 = ''
    var_34 = 'secure_project/file.txt'
    var_35 = 'secret'
    var_36 = 'password'
    var_37 = False
    var_38 = 'wrongpassword'
    var_39 = False
    var_40 = True
    var_41 = 'pathlib.Path.expanduser'
    var_42 = '~'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 0
    var_6 = 'https://example.com/repo.zip'
    var_7 = True
    var_8 = 'project_name'
    var_9 = 'local_repo.zip'
    var_10 = 'my_project/'
    var_11 = ''
    var_12 = 'my_project/README.md'
    var_13 = 'test'
    var_14 = False
    var_15 = True
    var_16 = 'my_project'
    var_17 = 'empty.zip'
    var_18 = False
    var_19 = 'no_dir.zip'
    var_20 = 'file.txt'
    var_21 = 'content'
    var_22 = False
    var_23 = 'bad.zip'
    var_24 = 'not a zip file'
    var_25 = False
    var_26 = 'pwd_repo.zip'
    var_27 = 'secure_project/'
    var_28 = ''
    var_29 = 'secure_project/file.txt'
    var_30 = 'secret'
    var_31 = b'mypassword'
    var_32 = False
    var_33 = 'mypassword'
    var_34 = 'secure_project'
    var_35 = False
    var_36 = 'wrongpassword'
    var_37 = False
    var_38 = True
    var_39 = 'cached.zip'
    var_40 = 'cached_proj/'
    var_41 = ''
    var_42 = 'https://example.com/cached.zip'
    var_43 = True
    var_44 = False
    var_45 = 'cached_proj'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False

def test_case_0():
    var_0 = "Test unzip downloads from URL when file doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'Test unzip uses cached file when prompt_and_delete returns False.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = False

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip raises error with invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'correct_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = 'wrong_password'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected with no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = True

def test_case_0():
    var_0 = 'Test unzip prompts user for password.'



# Parsed testcases at query #6
#--------------------------


import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_dir.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'cache'
    var_18 = 'url_test.zip'
    var_19 = 'url_project/'
    var_20 = ''
    var_21 = 'url_project/test.txt'
    var_22 = 'content'
    var_23 = 'rb'
    var_24 = 'requests.get'
    var_25 = 'http://example.com/url_test.zip'
    var_26 = True
    var_27 = 'protected.zip'
    var_28 = 'secret'
    var_29 = 'utf-8'
    var_30 = module_0.encode(var_29)
    var_31 = 'secure_project/'
    var_32 = ''
    var_33 = 'secure_project/file.txt'
    var_34 = 'secret content'
    var_35 = False
    var_36 = 'wrongpassword'
    var_37 = False
    var_38 = True
    var_39 = 'cookiecutter.prompt.prompt_and_delete'
    var_40 = module_1.patch(var_39)
    var_41 = 'http://example.com/cached.zip'



# Parsed testcases at query #7
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = b'not a zip file'
    var_16 = False
    var_17 = 'url_test.zip'
    var_18 = 'url_project/'
    var_19 = ''
    var_20 = 'url_project/file.txt'
    var_21 = 'content'
    var_22 = 'http://example.com/url_test.zip'
    var_23 = True
    var_24 = 'protected.zip'
    var_25 = 'test_password'
    var_26 = 'utf-8'
    var_27 = module_0.encode(var_26)
    var_28 = 'protected_project/'
    var_29 = ''
    var_30 = 'protected_project/file.txt'
    var_31 = 'content'
    var_32 = False
    var_33 = False
    var_34 = True
    var_35 = False
    var_36 = False
    var_37 = 'wrong_password'
    var_38 = 'http://example.com/cached.zip'
    var_39 = True
    var_40 = False



# Parsed testcases at query #8
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'https://example.com/repo.zip'
    var_9 = 'repo.zip'
    var_10 = 'my_project/'
    var_11 = ''
    var_12 = 'my_project/file.txt'
    var_13 = 'content'
    var_14 = 'requests.get'
    var_15 = 'cookiecutter.prompt.prompt_and_delete'
    var_16 = True
    var_17 = module_0.patch(var_15)
    var_18 = 'empty.zip'
    var_19 = False
    var_20 = 'no_dir.zip'
    var_21 = 'file.txt'
    var_22 = 'content'
    var_23 = False
    var_24 = 'invalid.zip'
    var_25 = 'This is not a zip file'
    var_26 = False
    var_27 = 'protected.zip'
    var_28 = 'secure_project/'
    var_29 = ''
    var_30 = 'secure_project/file.txt'
    var_31 = 'content'
    var_32 = b'mypassword'
    var_33 = 'mypassword'
    var_34 = False
    var_35 = 'wrongpassword'
    var_36 = False
    var_37 = True



# Parsed testcases at query #9
#--------------------------


import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'
    var_10 = 'file.txt'
    var_11 = 'empty.zip'
    var_12 = False
    var_13 = 'invalid.zip'
    var_14 = 'file.txt'
    var_15 = 'content'
    var_16 = False
    var_17 = 'bad.zip'
    var_18 = 'not a zip file'
    var_19 = False
    var_20 = 'clone'
    var_21 = 'url_test.zip'
    var_22 = 'url_project/'
    var_23 = ''
    var_24 = 'url_project/readme.md'
    var_25 = 'readme content'
    var_26 = 'requests.get'
    var_27 = 'https://example.com/archive.zip'
    var_28 = 'url_project'
    var_29 = 'protected.zip'
    var_30 = 'test_password'
    var_31 = 'secure_project/'
    var_32 = ''
    var_33 = 'utf-8'
    var_34 = module_0.encode(var_33)
    var_35 = 'secure_project/secret.txt'
    var_36 = 'secret'
    var_37 = 'secure_project'
    var_38 = False
    var_39 = True
    var_40 = 'wrong_password'
    var_41 = False
    var_42 = True
    var_43 = 'cache_test.zip'
    var_44 = 'cached_project/'
    var_45 = ''
    var_46 = 'clone2'
    var_47 = 'archive.zip'
    var_48 = 'cookiecutter.zipfile.prompt_and_delete'
    var_49 = module_1.patch(var_48)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'notazip.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = 'url_test.zip'
    var_19 = 'remote_project/'
    var_20 = ''
    var_21 = 'remote_project/file.txt'
    var_22 = 'remote content'
    var_23 = 'https://example.com/remote_project.zip'
    var_24 = True
    var_25 = 'protected.zip'
    var_26 = 'secure_project/'
    var_27 = ''
    var_28 = 'secure_project/secret.txt'
    var_29 = 'secret'
    var_30 = 'secure_project/'
    var_31 = 'secure_project/secret.txt'
    var_32 = 'Bad password'
    var_33 = None
    var_34 = False
    var_35 = 'correct_password'
    var_36 = 'project/'
    var_37 = 'project/file.txt'
    var_38 = 'Bad password'
    var_39 = False
    var_40 = True
    var_41 = 'project/'
    var_42 = 'project/file.txt'
    var_43 = 'Bad password'
    var_44 = False



# Parsed testcases at query #11
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'invalid.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'clone'
    var_18 = b'PK\x03\x04'
    var_19 = [var_18]
    var_20 = 'project/'
    var_21 = 'project/file.txt'
    var_22 = 'http://example.com/repo.zip'
    var_23 = True
    var_24 = 'protected.zip'
    var_25 = 'test_password'
    var_26 = 'project/'
    var_27 = ''
    var_28 = 'project/file.txt'
    var_29 = 'content'
    var_30 = 'utf-8'
    var_31 = module_0.encode(var_30)
    var_32 = 'Bad password'
    var_33 = None
    var_34 = 'project/'
    var_35 = 'project/file.txt'
    var_36 = False
    var_37 = 'Bad password'
    var_38 = 'project/'
    var_39 = 'project/file.txt'
    var_40 = False
    var_41 = True
    var_42 = 'Bad password'
    var_43 = 'project/'
    var_44 = 'project/file.txt'
    var_45 = False
    var_46 = 'wrong_password'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'zip_files'
    var_2 = 'valid.zip'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'bad.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'protected.zip'
    var_18 = 'secure_project/'
    var_19 = ''
    var_20 = 'secure_project/file.txt'
    var_21 = 'secret'
    var_22 = b'test123'
    var_23 = 'secure_project/'
    var_24 = ''
    var_25 = b'test123'
    var_26 = 'secure_project/file.txt'
    var_27 = 'secret'
    var_28 = 'test123'
    var_29 = False
    var_30 = 'wrong'
    var_31 = 'Response'
    var_32 = ()
    var_33 = 'iter_content'
    var_34 = b'test data'
    var_35 = [var_34]
    var_36 = lambda self, chunk_size: var_35
    var_37 = {var_33: var_36}
    var_38 = 'get'
    var_39 = 'cookiecutter.repository.prompt_and_delete'
    var_40 = True
    var_41 = lambda x, no_input: var_40
    var_42 = 'from_url.zip'
    var_43 = 'url_project/'
    var_44 = ''
    var_45 = 'url_project/file.txt'
    var_46 = 'from url'
    var_47 = 1024
    var_48 = 'http://example.com/repo.zip'



# Parsed testcases at query #13
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'empty.zip'
    var_8 = False
    var_9 = 'no_dir.zip'
    var_10 = 'file.txt'
    var_11 = 'content'
    var_12 = False
    var_13 = 'bad.zip'
    var_14 = b'not a zip file'
    var_15 = False
    var_16 = 'url_test.zip'
    var_17 = 'url_project/'
    var_18 = ''
    var_19 = 'url_project/file.txt'
    var_20 = 'content'
    var_21 = 'requests.get'
    var_22 = 'cookiecutter.prompt.prompt_and_delete'
    var_23 = True
    var_24 = module_0.patch(var_22)
    var_25 = 'http://example.com/url_test.zip'
    var_26 = 'protected.zip'
    var_27 = b'secret'
    var_28 = 'pwd_project/'
    var_29 = ''
    var_30 = 'pwd_project/file.txt'
    var_31 = 'content'
    var_32 = 'secret'
    var_33 = False
    var_34 = 'wrong'
    var_35 = False
    var_36 = True
    var_37 = 'test2.zip'
    var_38 = 'project2/'
    var_39 = ''
    var_40 = 'project2/file.txt'
    var_41 = 'content'
    var_42 = 'cookiecutter.utils.make_sure_path_exists'
    var_43 = module_0.patch(var_42)
    var_44 = '~/test'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip with empty zip file raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = 'not a zip file'
    var_4 = False

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip downloads from URL correctly.'
    var_1 = 'clone'
    var_2 = module_0.BytesIO()
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = 'https://example.com/repo.zip'
    var_8 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project-name/'
    var_5 = ''
    var_6 = 'project-name/file.txt'
    var_7 = 'content'
    var_8 = 'utf-8'
    var_9 = module_0.encode(var_8)
    var_10 = False

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project-name/'
    var_5 = ''
    var_6 = 'project-name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'wrong_password'

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'Test unzip with cached file and no_input=True skips download.'
    var_1 = 'clone'
    var_2 = 'repo.zip'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = 'https://example.com/repo.zip'
    var_8 = True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'nonexistent'
    var_3 = 'clone'
    var_4 = 'project-name/'
    var_5 = ''
    var_6 = 'project-name/file.txt'
    var_7 = 'content'
    var_8 = False



# Parsed testcases at query #15
#--------------------------


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip function with various scenarios.'
    var_1 = b'chunk1'
    var_2 = b'chunk2'
    var_3 = 'project-name/'
    var_4 = 'project-name/file.txt'
    var_5 = 'http://example.com/repo.zip'
    var_6 = True
    var_7 = module_0.unzip(var_5, var_6)
    assert var_7 == '/tmp/test/project-name'
    var_8 = 100
    var_9 = '/tmp/test'
    var_10 = 'my-project/'
    var_11 = 'my-project/file.txt'
    var_12 = '/local/path/repo.zip'
    var_13 = False
    var_14 = module_0.unzip(var_12, var_13)
    assert var_14 == '/tmp/test/my-project'
    var_15 = '/local/repo.zip'
    var_16 = False
    var_17 = module_0.unzip(var_15, var_16)
    var_18 = 'file.txt'
    var_19 = 'other.txt'
    var_20 = '/local/repo.zip'
    var_21 = False
    var_22 = module_0.unzip(var_20, var_21)
    var_23 = 'secure-project/'
    var_24 = 'secure-project/file.txt'
    var_25 = 'Bad password'
    var_26 = None
    var_27 = '/local/repo.zip'
    var_28 = False
    var_29 = 'mypassword'
    var_30 = module_0.unzip(var_27, var_28, password=var_29)
    assert var_30 == '/tmp/test/secure-project'
    var_31 = 'project/'
    var_32 = 'project/file.txt'
    var_33 = 'Bad password'
    var_34 = '/local/repo.zip'
    var_35 = False
    var_36 = 'wrongpassword'
    var_37 = module_0.unzip(var_34, var_35, password=var_36)
    var_38 = 'Not a valid zip'
    var_39 = '/local/repo.zip'
    var_40 = False
    var_41 = module_0.unzip(var_39, var_40)
    var_42 = 'project/'
    var_43 = 'project/file.txt'
    var_44 = '/local/repo.zip'
    var_45 = False
    var_46 = './custom_dir'
    var_47 = module_0.unzip(var_44, var_45, var_46)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'
    var_10 = 'empty.zip'
    var_11 = False
    var_12 = True
    var_13 = 'no_toplevel.zip'
    var_14 = 'file.txt'
    var_15 = 'content'
    var_16 = False
    var_17 = True
    var_18 = 'invalid.zip'
    var_19 = 'not a zip file'
    var_20 = False
    var_21 = True
    var_22 = b'test'
    var_23 = [var_22]
    var_24 = 'from_url.zip'
    var_25 = 'remote_project/'
    var_26 = ''
    var_27 = 'remote_project/file.txt'
    var_28 = 'content'
    var_29 = [var_25]
    var_30 = 'http://example.com/archive.zip'
    var_31 = True
    var_32 = 'protected.zip'
    var_33 = b'password'
    var_34 = 'secure_project/'
    var_35 = ''
    var_36 = 'secure_project/file.txt'
    var_37 = 'content'
    var_38 = 'password'
    var_39 = 'secure_project'
    var_40 = False
    var_41 = True
    var_42 = 'wrongpassword'
    var_43 = False
    var_44 = True
    var_45 = 'cached.zip'
    var_46 = 'cached_project/'
    var_47 = ''
    var_48 = 'cached_project/file.txt'
    var_49 = 'content'
    var_50 = 'http://example.com/cached.zip'
    var_51 = True
    var_52 = False



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'zips'
    var_2 = 'extract'
    var_3 = 'test.zip'
    var_4 = 'project-name/'
    var_5 = ''
    var_6 = 'project-name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = 'bad.zip'
    var_12 = 'file.txt'
    var_13 = 'content'
    var_14 = False
    var_15 = 'invalid.zip'
    var_16 = 'not a zip file'
    var_17 = False
    var_18 = 'remote.zip'
    var_19 = 'remote-project/'
    var_20 = ''
    var_21 = 'remote-project/file.txt'
    var_22 = 'content'
    var_23 = 'rb'
    var_24 = 'https://example.com/remote.zip'
    var_25 = True
    var_26 = 'protected.zip'
    var_27 = b'secret'
    var_28 = 'protected-project/'
    var_29 = ''
    var_30 = 'protected-project/file.txt'
    var_31 = 'content'
    var_32 = 'secret'
    var_33 = False
    var_34 = 'wrongpassword'
    var_35 = False
    var_36 = True
    var_37 = 'https://example.com/cached.zip'
    var_38 = True
    var_39 = False



# Parsed testcases at query #18
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'notoplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'invalid.zip'
    var_15 = b'not a zip file'
    var_16 = False
    var_17 = 'url_test.zip'
    var_18 = 'remote_project/'
    var_19 = ''
    var_20 = 'remote_project/file.txt'
    var_21 = 'remote content'
    var_22 = [var_18]
    var_23 = 'https://example.com/project.zip'
    var_24 = True
    var_25 = 'protected.zip'
    var_26 = b'test123'
    var_27 = 'secure_project/'
    var_28 = ''
    var_29 = 'secure_project/secret.txt'
    var_30 = 'secret'
    var_31 = 'test123'
    var_32 = False
    var_33 = 'wrongpassword'
    var_34 = False
    var_35 = True
    var_36 = 'new_clone_dir'
    var_37 = module_0.exists()
    var_38 = module_0.exists()
    var_39 = 'cached.zip'
    var_40 = 'cached_project/'
    var_41 = ''
    var_42 = 'https://example.com/cached.zip'
    var_43 = True
    var_44 = False



# Parsed testcases at query #19
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'empty.zip'
    var_9 = False
    var_10 = 'no_toplevel.zip'
    var_11 = 'file.txt'
    var_12 = 'content'
    var_13 = False
    var_14 = 'bad.zip'
    var_15 = 'not a zip file'
    var_16 = False
    var_17 = 'valid_url.zip'
    var_18 = 'remote_project/'
    var_19 = ''
    var_20 = 'remote_project/setup.py'
    var_21 = 'setup code'
    var_22 = [var_18]
    var_23 = 'http://example.com/project.zip'
    var_24 = True
    var_25 = 'protected.zip'
    var_26 = 'secure_project/'
    var_27 = ''
    var_28 = 'secure_project/file.txt'
    var_29 = 'secret'
    var_30 = b'mypassword'
    var_31 = 'mypassword'
    var_32 = False
    var_33 = 'wrongpassword'
    var_34 = 'project/'
    var_35 = 'project/file.txt'
    var_36 = 'Bad password'
    var_37 = False
    var_38 = True
    var_39 = 'new_dir'
    var_40 = module_0.exists()
    var_41 = module_0.exists()



# Parsed testcases at query #20
#--------------------------


import _io as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'clone'
    var_2 = module_0.BytesIO()
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 0
    var_8 = 'test.zip'
    var_9 = 'requests.get'
    var_10 = 'cookiecutter.zipfile.prompt_and_delete'
    var_11 = True
    var_12 = module_1.patch(var_10)
    var_13 = 'https://example.com/test.zip'
    var_14 = 'test_project'
    var_15 = False
    var_16 = module_0.BytesIO()
    var_17 = 'empty.zip'
    var_18 = False
    var_19 = module_0.BytesIO()
    var_20 = 'file.txt'
    var_21 = 'content'
    var_22 = 'nodir.zip'
    var_23 = False
    var_24 = 'invalid.zip'
    var_25 = b'not a zip file'
    var_26 = False
    var_27 = module_0.BytesIO()
    var_28 = 'protected_project/'
    var_29 = ''
    var_30 = 'protected_project/file.txt'
    var_31 = 'content'
    var_32 = 8
    var_33 = 'protected.zip'
    var_34 = 'protected_project/'
    var_35 = 'protected_project/file.txt'
    var_36 = 'cookiecutter.zipfile.ZipFile'
    var_37 = False
    var_38 = 'correct_password'
    var_39 = 'protected_project'
    var_40 = ()
    var_41 = 'Bad password'
    var_42 = False
    var_43 = 'wrong_password'



# Parsed testcases at query #21
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test the unzip function with various scenarios.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'empty.zip'
    var_10 = False
    var_11 = True
    var_12 = 'no_toplevel.zip'
    var_13 = 'file.txt'
    var_14 = 'content'
    var_15 = False
    var_16 = True
    var_17 = 'invalid.zip'
    var_18 = 'not a zip file'
    var_19 = False
    var_20 = True
    var_21 = 'clone'
    var_22 = b'test_content'
    var_23 = [var_22]
    var_24 = 'requests.get'
    var_25 = 'repo.zip'
    var_26 = 'repo/'
    var_27 = ''
    var_28 = 'repo/file.txt'
    var_29 = 'content'
    var_30 = 'http://example.com/repo.zip'
    var_31 = 'protected.zip'
    var_32 = 'test_password'
    var_33 = 'utf-8'
    var_34 = module_0.encode(var_33)
    var_35 = 'secure_project/'
    var_36 = ''
    var_37 = 'secure_project/file.txt'
    var_38 = 'secret'
    var_39 = False
    var_40 = True
    var_41 = 'wrong_password'
    var_42 = False
    var_43 = True



