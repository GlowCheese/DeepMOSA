####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 2/6 statements.
# Partially parsed test_unzip_url. Retrieved 3/9 statements.
# Partially parsed test_unzip_password_protected. Retrieved 2/8 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 2/7 statements.
# Partially parsed test_unzip_no_top_level_dir. Retrieved 2/7 statements.
# Partially parsed test_unzip_invalid_zip. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = False

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = b'test'
    var_2 = True
    var_3 = bool(var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = False
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'secret'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, password=var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = '/tmp/custom_dir'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = bool(var_1 in var_3)
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_chunk_is_not_empty. Retrieved 2/9 statements.


def test_case_0():
    var_0 = b'some data'
    var_1 = b''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_path_exists. Retrieved 6/10 statements.


import zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Path(var_0)
    var_2 = True
    var_3 = 'test.zip'
    var_4 = var_1 / var_3
    var_5 = module_1.exists(var_4)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_predicate_at_line_39_evaluates_to_false. Retrieved 6/20 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/file.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_predicate_at_line_36_evaluates_to_false. Retrieved 10/15 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'test.zip'
    var_5 = [var_4]
    var_6 = True
    var_7 = 'test'
    var_8 = False
    var_9 = lambda path, no_input: var_8
    var_10 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_skips_empty_chunks. Retrieved 15/22 statements.


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b''
    var_4 = b'data'
    var_5 = [var_3, var_4, var_3]
    var_6 = lambda self, chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockFile'
    var_10 = ()
    var_11 = 'write'
    var_12 = None
    var_13 = lambda self, chunk: var_12
    var_14 = {var_11: var_13}
    var_15 = [var_9, var_10, var_14]
    var_16 = 1024



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_does_not_download_when_file_exists_and_user_chooses_to_reuse. Retrieved 10/18 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = False
    var_4 = None
    var_5 = True
    var_6 = False
    var_7 = lambda path, no_input: var_6
    var_8 = lambda question, default: var_5
    var_9 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_does_not_download_when_file_exists_and_user_chooses_to_reuse. Retrieved 10/20 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = True
    var_6 = False
    var_7 = lambda path, no_input: var_6
    var_8 = 'Should not download when reusing existing file'
    var_9 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 14/19 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 3/7 statements.
# Partially parsed test_unzip_no_top_level_dir. Retrieved 5/9 statements.
# Partially parsed test_unzip_password_protected. Retrieved 17/23 statements.
# Partially parsed test_unzip_invalid_zip. Retrieved 4/8 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project'
    var_2 = f'{var_1}/'
    var_3 = ''
    var_4 = f'{var_1}/file.txt'
    var_5 = 'test content'
    var_6 = False
    var_7 = module_0.unzip(var_0, var_6)
    var_8 = module_1.exists(var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = module_1.isdir(var_7)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = 'file.txt'
    var_13 = [var_12]
    var_14 = module_1.exists(var_5)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = module_2.dirname(var_7)
    var_17 = module_3.rmtree(var_16)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import email._encoded_words as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2
import posixpath as module_3
import cookiecutter.utils as module_4

def test_case_0():
    var_0 = 'password.zip'
    var_1 = 'test_project'
    var_2 = 'secret'
    var_3 = f'{var_1}/'
    var_4 = ''
    var_5 = f'{var_1}/file.txt'
    var_6 = 'test content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = module_1.unzip(var_0, var_9, password=var_2)
    var_11 = module_2.exists(var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = module_2.isdir(var_10)
    var_14 = bool(var_13)
    assert var_14 is True
    var_15 = 'file.txt'
    var_16 = [var_15]
    var_17 = module_2.exists(var_6)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = module_3.dirname(var_10)
    var_20 = module_4.rmtree(var_19)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_empty_zip_file_raises_exception. Retrieved 1/12 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 6/7 statements.
# Partially parsed test_unzip_url_success. Retrieved 6/7 statements.
# Partially parsed test_unzip_password_protected_success. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'secret'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/password_protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/password_protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/bad.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'secret'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, password=var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = True
    var_3 = module_0.unzip(var_0, var_1, no_input=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'existing.zip'
    var_1 = False
    var_2 = True
    var_3 = module_0.unzip(var_0, var_1, no_input=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 7/25 statements.
# Partially parsed test_unzip_url. Retrieved 3/15 statements.
# Partially parsed test_unzip_password_protected. Retrieved 9/30 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 2/14 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 4/16 statements.
# Partially parsed test_unzip_invalid_zip. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'testdir/'
    var_2 = ''
    var_3 = 'testdir/testfile.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'testfile.txt'

def test_case_0():
    var_0 = b'test content'
    var_1 = 'http://example.com/test.zip'
    var_2 = True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'testdir/'
    var_2 = ''
    var_3 = 'testdir/testfile.txt'
    var_4 = 'test content'
    var_5 = b'password'
    var_6 = False
    var_7 = 'password'
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = bool(var_5)
    assert var_9 is True
    var_10 = 'testfile.txt'

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'testfile.txt'
    var_2 = 'test content'
    var_3 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = b'invalid zip content'
    var_2 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_url. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = True
    var_3 = module_0.unzip(var_0, var_1, no_input=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = False
    var_3 = module_0.unzip(var_0, var_1, no_input=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/protected.zip'
    var_1 = False
    var_2 = 'password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/protected.zip'
    var_1 = False
    var_2 = 'wrongpassword'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/no_top_level.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 14/19 statements.
# Partially parsed test_unzip_url_success. Retrieved 16/29 statements.
# Partially parsed test_unzip_empty_zip_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_unzip_no_top_level_dir_raises_exception. Retrieved 5/9 statements.
# Partially parsed test_unzip_password_protected_success. Retrieved 18/27 statements.
# Partially parsed test_unzip_invalid_password_raises_exception. Retrieved 12/18 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project'
    var_2 = f'{var_1}/'
    var_3 = ''
    var_4 = f'{var_1}/file.txt'
    var_5 = 'test content'
    var_6 = False
    var_7 = module_0.unzip(var_0, var_6)
    var_8 = module_1.exists(var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = module_1.isdir(var_7)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = 'file.txt'
    var_13 = [var_12]
    var_14 = module_1.exists(var_5)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = module_2.dirname(var_7)
    var_17 = module_3.rmtree(var_16)

import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = 'test.zip'
    var_2 = 'test_project'
    var_3 = 'get'
    var_4 = f'{var_2}/'
    var_5 = ''
    var_6 = f'{var_2}/file.txt'
    var_7 = 'test content'
    var_8 = True
    var_9 = '.'
    var_10 = module_0.unzip(var_0, var_8, var_9)
    var_11 = module_1.exists(var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = module_1.isdir(var_10)
    var_14 = bool(var_13)
    assert var_14 is True
    var_15 = 'file.txt'
    var_16 = [var_15]
    var_17 = module_2.dirname(var_10)
    var_18 = module_3.rmtree(var_17)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = 'file.txt'
    var_2 = 'test content'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3)
    var_5 = bool(False)
    assert var_5 is True

import email._encoded_words as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2
import posixpath as module_3
import cookiecutter.utils as module_4

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'protected_project'
    var_2 = 'secret'
    var_3 = 'cookiecutter.prompt.read_repo_password'
    var_4 = f'{var_1}/'
    var_5 = ''
    var_6 = f'{var_1}/file.txt'
    var_7 = 'test content'
    var_8 = 'utf-8'
    var_9 = module_0.encode(var_8)
    var_10 = False
    var_11 = module_1.unzip(var_0, var_10, password=var_2)
    var_12 = module_2.exists(var_11)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = module_2.isdir(var_11)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'file.txt'
    var_17 = [var_16]
    var_18 = module_2.exists(var_8)
    var_19 = bool(var_18)
    assert var_19 is True
    var_20 = module_3.dirname(var_11)
    var_21 = module_4.rmtree(var_20)

import email._encoded_words as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'protected_project'
    var_2 = 'secret'
    var_3 = f'{var_1}/'
    var_4 = ''
    var_5 = f'{var_1}/file.txt'
    var_6 = 'test content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = 'wrong'
    var_11 = module_1.unzip(var_0, var_9, password=var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_evaluates_to_false_when_file_exists_and_no_input_is_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = 'repo.zip'
    var_5 = [var_4]
    var_6 = True
    var_7 = 'w'



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test-repo.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.isdir(var_5)
    var_9 = bool(var_8)
    assert var_9 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test-repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.isdir(var_5)
    var_9 = bool(var_8)
    assert var_9 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/protected-repo.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'secret'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.isdir(var_5)
    var_9 = bool(var_8)
    assert var_9 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/invalid-repo.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/empty-repo.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/no-top-level-repo.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #9
#--------------------------




import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = module_0.abspath(var_0)
    var_3 = module_1.unzip(var_0, var_1)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.zipfile as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'local_file.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.abspath(var_0)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_existing_zip_path_with_no_input. Retrieved 10/13 statements.


import cookiecutter.utils as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = '/tmp/clone_dir'
    var_2 = module_0.make_sure_path_exists(var_1)
    var_3 = 1
    var_4 = '/'
    var_5 = var_0.rsplit(var_4, var_3)[var_3]
    var_6 = [var_5]
    var_7 = 'dummy content'
    var_8 = True
    var_9 = True
    var_10 = module_1.unzip(var_0, var_8, var_1, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 11/24 statements.
# Partially parsed test_unzip_url. Retrieved 12/19 statements.
# Partially parsed test_unzip_password_protected. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip. Retrieved 3/10 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 3/11 statements.
# Partially parsed test_unzip_no_top_level_dir. Retrieved 4/12 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'test content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.isdir(var_5)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'file.txt'
    var_11 = [var_10]
    var_12 = module_2.dirname(var_5)
    var_13 = module_3.rmtree(var_12)

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = b'test'
    var_1 = 'requests.get'
    var_2 = 'zipfile.ZipFile'
    var_3 = True
    var_4 = 'autospec'
    var_5 = {var_4: var_3}
    var_6 = module_0.patch(var_2, **var_5)
    var_7 = 'tempfile.mkdtemp'
    var_8 = '/tmp/test'
    var_9 = 'return_value'
    var_10 = {var_9: var_8}
    var_11 = module_0.patch(var_7, **var_10)
    var_12 = 'cookiecutter.prompt.prompt_and_delete'
    var_13 = 'return_value'
    var_14 = {var_13: var_3}
    var_15 = module_0.patch(var_12, **var_14)
    var_16 = 'http://example.com/test.zip'
    var_17 = module_1.unzip(var_16, var_3)
    assert var_17 == '/tmp/test/testdir'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'testdir/'
    var_1 = []
    var_2 = None
    var_3 = 'zipfile.ZipFile'
    var_4 = 'cookiecutter.prompt.read_repo_password'
    var_5 = 'password'
    var_6 = 'return_value'
    var_7 = {var_6: var_5}
    var_8 = module_0.patch(var_4, **var_7)
    var_9 = False
    var_10 = 'password'
    var_11 = module_1.unzip(var_0, var_9, password=var_10)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'invalid zip content'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'zipfile.ZipFile'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'zipfile.ZipFile'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'top-level directory'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'bad_structure.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'secret'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, password=var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/existing.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = module_1.isdir(var_2)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_does_not_download_when_file_exists_and_user_chooses_to_reuse. Retrieved 6/10 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_local_zipfile. Retrieved 5/6 statements.
# Partially parsed test_unzip_url_zipfile. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password. Retrieved 6/7 statements.
# Partially parsed test_unzip_no_input. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_with_non_empty_zip. Retrieved 10/13 statements.


import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5)
    var_7 = module_1.ZipFile(var_0)
    var_8 = var_7.namelist()
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_empty_zip_file_raises_exception. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #20
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True



