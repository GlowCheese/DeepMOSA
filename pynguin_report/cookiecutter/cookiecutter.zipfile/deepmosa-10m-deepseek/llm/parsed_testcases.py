####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_url. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password. Retrieved 6/7 statements.
# Partially parsed test_unzip_no_input. Retrieved 6/7 statements.


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
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_downloads_zipfile_when_not_exists. Retrieved 24/37 statements.


import zipfile as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = False
    var_4 = None
    var_5 = module_0.Path(var_2)
    var_6 = True
    var_7 = 'MockResponse'
    var_8 = ()
    var_9 = 'iter_content'
    var_10 = 'status_code'
    var_11 = b'test'
    var_12 = [var_11]
    var_13 = lambda self, chunk_size: var_12
    var_14 = 200
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = [var_7, var_8, var_15]
    var_17 = 'MockFile'
    var_18 = ()
    var_19 = 'write'
    var_20 = None
    var_21 = lambda self, data: var_20
    var_22 = {var_19: var_21}
    var_23 = [var_17, var_18, var_22]
    var_24 = False
    var_25 = module_1.unzip(var_0, var_1, var_2, var_3, var_4)
    var_26 = bool(True)
    assert var_26 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_existing_zip_path_and_no_input_true. Retrieved 9/14 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = True
    var_4 = None
    var_5 = True
    var_6 = 'repo.zip'
    var_7 = [var_6]
    var_8 = 'test'
    var_9 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'non_existing_file.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_does_not_download_when_prompt_rejects_deletion_and_reuse. Retrieved 10/21 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = False
    var_4 = True
    var_5 = False
    var_6 = lambda path, no_input: var_5
    var_7 = lambda question, default: var_5
    var_8 = None
    var_9 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_no_download_when_exists_and_reuse. Retrieved 13/18 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = False
    var_4 = None
    var_5 = 'repo.zip'
    var_6 = [var_5]
    var_7 = True
    var_8 = 'dummy content'
    var_9 = 'no'
    var_10 = lambda *args: var_9
    var_11 = False
    var_12 = lambda *args: var_11
    var_13 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test_data/test_zip.zip'
    var_1 = False
    var_2 = 'tests/test_tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test_data/protected_zip.zip'
    var_1 = False
    var_2 = 'tests/test_tmp'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = module_1.isdir(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test_zip.zip'
    var_1 = True
    var_2 = 'tests/test_tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test_zip.zip'
    var_1 = True
    var_2 = 'tests/test_tmp'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = module_1.isdir(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/empty_zip.zip'
    var_1 = False
    var_2 = 'tests/test_tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/invalid_zip.zip'
    var_1 = False
    var_2 = 'tests/test_tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/protected_zip.zip'
    var_1 = False
    var_2 = 'tests/test_tmp'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_valid_zipfile. Retrieved 13/19 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'valid.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = False
    assert var_5 == 'test content'
    var_6 = module_0.unzip(var_0, var_5)
    var_7 = module_1.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = module_1.isdir(var_6)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = 'test_file.txt'
    var_12 = [var_11]
    var_13 = module_1.exists(var_4)
    var_14 = bool(var_13)
    assert var_14 is True
    var_15 = module_2.dirname(var_6)
    var_16 = module_3.rmtree(var_15)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_False. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = False
    var_4 = None
    var_5 = True
    var_6 = '/'
    var_7 = var_0.rsplit(var_6, var_5)[var_5]
    var_8 = [var_7]
    var_9 = 'dummy content'



# Parsed testcases at query #12
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test.zip'
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
    var_0 = 'https://example.com/test.zip'
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
    var_0 = 'tests/protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/no_top_level.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 13/22 statements.
# Partially parsed test_unzip_url. Retrieved 21/33 statements.
# Partially parsed test_unzip_password_protected. Retrieved 17/26 statements.
# Partially parsed test_unzip_invalid_zip. Retrieved 3/9 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 1/9 statements.
# Partially parsed test_unzip_no_top_level_dir. Retrieved 4/11 statements.


import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'test content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)
    var_6 = module_1.Path(var_5)
    var_7 = var_6.exists()
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = module_1.Path(var_5)
    var_10 = var_9.is_dir()
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = module_1.Path(var_5)
    var_13 = 'file.txt'
    var_14 = var_12 / var_13

import requests.api as module_0
import zipfile as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = b'test content'
    var_1 = 'requests.get'
    var_2 = 'cookiecutter.prompt.prompt_and_delete'
    var_3 = True
    var_4 = 'return_value'
    var_5 = {var_4: var_3}
    var_6 = module_0.patch(var_2, **var_5)
    var_7 = 'testdir/'
    var_8 = ''
    var_9 = 'testdir/file.txt'
    var_10 = 'test content'
    var_11 = 'builtins.open'
    var_12 = {}
    var_13 = module_0.patch(var_11, var_8, **var_12)
    var_14 = 'zipfile.ZipFile'
    var_15 = module_1.ZipFile(var_10)
    var_16 = 'return_value'
    var_17 = {var_16: var_15}
    var_18 = module_0.patch(var_14, **var_17)
    var_19 = 'http://example.com/test.zip'
    var_20 = True
    var_21 = module_2.unzip(var_19, var_20)
    var_22 = module_1.Path(var_21)
    var_23 = var_22.exists()
    var_24 = bool(var_23)
    assert var_24 is True
    var_25 = module_1.Path(var_21)
    var_26 = var_25.is_dir()
    var_27 = bool(var_26)
    assert var_27 is True

import requests.api as module_0
import cookiecutter.zipfile as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'test content'
    var_4 = b'password'
    var_5 = 'cookiecutter.prompt.prompt_and_delete'
    var_6 = True
    var_7 = 'return_value'
    var_8 = {var_7: var_6}
    var_9 = module_0.patch(var_5, **var_8)
    var_10 = 'cookiecutter.prompt.read_repo_password'
    var_11 = 'password'
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = False
    var_16 = module_1.unzip(var_4, var_15, password=var_11)
    var_17 = module_2.Path(var_16)
    var_18 = var_17.exists()
    var_19 = bool(var_18)
    assert var_19 is True
    var_20 = module_2.Path(var_16)
    var_21 = var_20.is_dir()
    var_22 = bool(var_21)
    assert var_22 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'invalid zip content'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not a valid zip archive'

def test_case_0():
    var_0 = False
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'is empty'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'test content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'does not include a top-level directory'



# Parsed testcases at query #3
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
    var_0 = 'http://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

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

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_does_not_download_when_prompt_and_delete_returns_false. Retrieved 6/12 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_non_empty_zip_file. Retrieved 6/17 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test_dir/'
    var_1 = ''
    var_2 = 'test_dir/test_file.txt'
    var_3 = 'test content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 14/20 statements.
# Partially parsed test_unzip_url_success. Retrieved 17/31 statements.
# Partially parsed test_unzip_empty_zip_raises_exception. Retrieved 3/8 statements.
# Partially parsed test_unzip_no_top_level_dir_raises_exception. Retrieved 5/10 statements.
# Partially parsed test_unzip_password_protected_success. Retrieved 19/28 statements.
# Partially parsed test_unzip_invalid_password_raises_exception. Retrieved 15/23 statements.


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
    var_5 = 'content'
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
    var_7 = 'content'
    var_8 = 'cookiecutter.zipfile.prompt_and_delete'
    var_9 = True
    var_10 = lambda *args, **kwargs: var_9
    var_11 = module_0.unzip(var_0, var_9)
    var_12 = module_1.exists(var_11)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = module_1.isdir(var_11)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'file.txt'
    var_17 = [var_16]
    var_18 = module_2.dirname(var_11)
    var_19 = module_3.rmtree(var_18)

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
    var_2 = 'content'
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
    var_3 = f'{var_1}/'
    var_4 = ''
    var_5 = f'{var_1}/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'cookiecutter.zipfile.read_repo_password'
    var_10 = lambda *args: var_2
    var_11 = False
    var_12 = None
    var_13 = module_1.unzip(var_0, var_11, password=var_12)
    var_14 = module_2.exists(var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = module_2.isdir(var_13)
    var_17 = bool(var_16)
    assert var_17 is True
    var_18 = 'file.txt'
    var_19 = [var_18]
    var_20 = module_3.dirname(var_13)
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
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'cookiecutter.zipfile.read_repo_password'
    var_10 = 'wrong'
    var_11 = lambda *args: var_10
    var_12 = False
    var_13 = None
    var_14 = module_1.unzip(var_0, var_12, password=var_13)
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = 'test.zip'



# Parsed testcases at query #9
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_local_zipfile. Retrieved 4/5 statements.
# Partially parsed test_unzip_url_zipfile. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_password. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_no_input. Retrieved 5/6 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)

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
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True



