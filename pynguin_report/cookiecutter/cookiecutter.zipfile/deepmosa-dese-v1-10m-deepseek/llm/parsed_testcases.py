####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 8/22 statements.
# Partially parsed test_unzip_url_success. Retrieved 14/32 statements.
# Partially parsed test_unzip_empty_zip_raises_exception. Retrieved 1/11 statements.
# Partially parsed test_unzip_no_top_level_dir_raises_exception. Retrieved 3/13 statements.
# Partially parsed test_unzip_password_protected_success. Retrieved 11/26 statements.
# Partially parsed test_unzip_invalid_password_raises_exception. Retrieved 7/19 statements.


import genericpath as module_0

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = module_0.exists()
    var_6 = 'file.txt'
    var_7 = module_0.exists()

import cookiecutter.zipfile as module_0
import zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'requests.get'
    var_1 = 'cookiecutter.zipfile.prompt_and_delete'
    var_2 = True
    var_3 = lambda *args, **kwargs: var_2
    var_4 = 'http://example.com/test.zip'
    var_5 = module_0.unzip(var_4, var_2)
    var_6 = module_1.Path(var_5)
    var_7 = var_6.exists()
    var_8 = module_1.Path(var_5)
    var_9 = var_8.is_dir()
    var_10 = module_1.Path(var_5)
    var_11 = 'file.txt'
    var_12 = var_10 / var_11
    var_13 = module_2.exists()

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'content'
    var_2 = False

import genericpath as module_0

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = False
    var_6 = 'password'
    var_7 = module_0.exists()
    var_8 = 'file.txt'
    var_9 = var_4 / var_8
    var_10 = module_0.exists()

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = False
    var_6 = 'wrongpassword'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_existing_zip_path_and_no_input. Retrieved 8/12 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = True
    var_4 = True
    var_5 = 'repo.zip'
    var_6 = 'test'
    var_7 = module_0.unzip(var_0, var_1, var_2, var_3)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_valid_url_and_existing_download. Retrieved 8/11 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test-data/valid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test-data/empty.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/test-data/protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test-data/protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'wrongpassword'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test-data/protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/valid.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/valid.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = 'valid.zip'
    var_6 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_7 = module_1.exists(var_6)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_false. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/clone_dir'
    var_3 = False
    var_4 = None
    var_5 = True
    var_6 = 'repo.zip'
    var_7 = 'dummy content'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_skip_empty_chunks. Retrieved 8/12 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b''
    var_1 = b'data'
    var_2 = 'http://example.com/test.zip'
    var_3 = True
    var_4 = '.'
    var_5 = False
    var_6 = None
    var_7 = module_0.unzip(var_2, var_3, var_4, var_5, var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 11/24 statements.
# Partially parsed test_unzip_url. Retrieved 17/30 statements.
# Partially parsed test_unzip_password_protected. Retrieved 20/33 statements.
# Partially parsed test_unzip_empty_repository. Retrieved 1/13 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 4/15 statements.


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
    var_7 = module_1.isdir(var_5)
    var_8 = 'file.txt'
    var_9 = module_2.dirname(var_5)
    var_10 = module_3.rmtree(var_9)

import _io as module_0
import zipfile as module_1
import requests.api as module_2
import cookiecutter.zipfile as module_3

def test_case_0():
    var_0 = b'test'
    var_1 = 'requests.get'
    var_2 = module_0.BytesIO()
    var_3 = 'testdir/'
    var_4 = ''
    var_5 = 'testdir/file.txt'
    var_6 = 'test content'
    var_7 = 0
    var_8 = 'builtins.open'
    var_9 = 'zipfile.ZipFile'
    var_10 = module_1.ZipFile(var_2)
    var_11 = module_2.patch(var_9)
    var_12 = 'cookiecutter.prompt.prompt_and_delete'
    var_13 = True
    var_14 = module_2.patch(var_12)
    var_15 = 'http://example.com/test.zip'
    var_16 = module_3.unzip(var_15, var_13)

import _io as module_0
import zipfile as module_1
import requests.api as module_2
import cookiecutter.zipfile as module_3

def test_case_0():
    var_0 = b'test'
    var_1 = 'requests.get'
    var_2 = module_0.BytesIO()
    var_3 = 'testdir/'
    var_4 = ''
    var_5 = 'testdir/file.txt'
    var_6 = 'test content'
    var_7 = 0
    var_8 = 'builtins.open'
    var_9 = 'zipfile.ZipFile'
    var_10 = module_1.ZipFile(var_2)
    var_11 = module_2.patch(var_9)
    var_12 = 'cookiecutter.prompt.prompt_and_delete'
    var_13 = True
    var_14 = module_2.patch(var_12)
    var_15 = 'cookiecutter.prompt.read_repo_password'
    var_16 = 'password'
    var_17 = module_2.patch(var_15)
    var_18 = 'http://example.com/test.zip'
    var_19 = module_3.unzip(var_18, var_13, password=var_16)

def test_case_0():
    var_0 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'test content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)



# Parsed testcases at query #7
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
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = 'wrongpassword'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_empty_zip_file_raises_exception. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(var_0)
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = 'tests/test_data/test.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = 'tests/temp'
    var_6 = module_2.rmtree(var_5)

import cookiecutter.zipfile as module_0
import genericpath as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = 'tests/temp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = 'tests/temp'
    var_6 = module_2.rmtree(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/empty.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/invalid.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = 'tests/test_data/protected.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = 'password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = 'tests/temp'
    var_7 = module_2.rmtree(var_6)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/protected.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/test_data/protected.zip'
    var_1 = False
    var_2 = 'tests/temp'
    var_3 = 'wrongpassword'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = True
    var_4 = None



# Parsed testcases at query #12
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/path/to/existing/file'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_creates_zipfile. Retrieved 11/22 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = True
    var_3 = 'http://example.com/test.zip'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.unzip(var_3, var_5, var_4, var_6)
    var_8 = 100
    var_9 = '/tmp/test.zip'
    var_10 = 'wb'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_download_false. Retrieved 10/16 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'test'
    var_8 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_9 = module_1.exists(var_8)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_zip_file_context_manager_releases_file_descriptor. Retrieved 4/12 statements.


import zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = module_0.Path(var_0)
    var_2 = 'test/'
    var_3 = ''



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 13/22 statements.
# Partially parsed test_unzip_empty_zip_raises_exception. Retrieved 1/10 statements.
# Partially parsed test_unzip_no_top_level_dir_raises_exception. Retrieved 4/12 statements.
# Partially parsed test_unzip_invalid_zip_raises_exception. Retrieved 3/10 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 15/25 statements.


import cookiecutter.zipfile as module_0
import zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)
    var_6 = module_1.Path(var_5)
    var_7 = var_6.exists()
    var_8 = module_1.Path(var_5)
    var_9 = var_8.name
    assert var_9 == 'testdir'
    var_10 = module_1.Path(var_5)
    var_11 = 'file.txt'
    var_12 = module_2.exists()

def test_case_0():
    var_0 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'invalid zip content'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'testdir/'
    var_1 = ''
    var_2 = 'testdir/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = False
    var_6 = 'password'
    var_7 = module_0.unzip(var_0, var_5, password=var_6)
    var_8 = module_1.Path(var_7)
    var_9 = var_8.exists()
    var_10 = module_1.Path(var_7)
    var_11 = var_10.name
    assert var_11 == 'testdir'
    var_12 = module_1.Path(var_7)
    var_13 = 'file.txt'
    var_14 = module_2.exists()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_predicate_evaluates_to_true_when_zip_path_exists. Retrieved 8/13 statements.


import zipfile as module_0

def test_case_0():
    var_0 = '.'
    var_1 = module_0.Path(var_0)
    var_2 = 'https://example.com/repo.zip'
    var_3 = 1
    var_4 = '/'
    var_5 = zip_uri.rsplit(var_4, var_3)[var_3]
    var_6 = True
    var_7 = 'test content'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/17 statements.
# Partially parsed test_unzip_url_file. Retrieved 9/18 statements.
# Partially parsed test_unzip_empty_file. Retrieved 2/9 statements.
# Partially parsed test_unzip_invalid_file. Retrieved 3/10 statements.
# Partially parsed test_unzip_password_protected_file. Retrieved 8/17 statements.
# Partially parsed test_unzip_password_protected_file_invalid_password. Retrieved 8/17 statements.


import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = False
    var_6 = 'test_file.txt'
    var_7 = module_0.isfile(var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = True
    var_6 = module_0.unzip(var_2, var_5)
    var_7 = module_1.exists(var_6)
    var_8 = 'test_file.txt'

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'invalid content'
    var_2 = False

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = b'password'
    var_6 = False
    var_7 = 'password'

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/test_file.txt'
    var_4 = 'test content'
    var_5 = b'password'
    var_6 = False
    var_7 = 'wrong_password'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_when_bad_zip_file. Retrieved 2/12 statements.


def test_case_0():
    var_0 = b'This is not a valid zip file'
    var_1 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_prompt_and_delete_returns_false_when_user_reuses_existing. Retrieved 3/10 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is False



# Parsed testcases at query #6
#--------------------------




import zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = module_0.ZipFile(var_0)
    var_2 = var_1.close()



# Parsed testcases at query #7
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_dir'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, var_1)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = 'test_dir'
    var_2 = True
    var_3 = module_0.unzip(var_0, var_2, var_1, var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_dir'
    var_2 = 'secret'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3, var_1, password=var_2)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'test_dir'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = 'test_dir'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'test_dir'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, var_1)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = b'some binary data'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_downloads_file_when_not_exists. Retrieved 23/34 statements.


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
    var_16 = 'MockFile'
    var_17 = ()
    var_18 = 'write'
    var_19 = None
    var_20 = lambda self, data: var_19
    var_21 = {var_18: var_20}
    var_22 = module_1.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b''
    var_4 = [var_3, var_3]
    var_5 = lambda self, chunk_size: var_4
    var_6 = {var_2: var_5}
    var_7 = 1024



# Parsed testcases at query #12
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_with_existing_zip_and_no_download. Retrieved 9/15 statements.


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
    var_8 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_non_empty_repository. Retrieved 11/14 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'non_empty_repo.zip'
    var_1 = 'http://example.com/non_empty_repo.zip'
    var_2 = True
    var_3 = '/path/to/clone'
    var_4 = False
    var_5 = None
    var_6 = 'file1.txt'
    var_7 = 'content'
    var_8 = 'file2.txt'
    var_9 = module_0.unzip(var_1, var_2, var_3, var_4, var_5)
    var_10 = module_1.exists(var_9)



# Parsed testcases at query #16
#--------------------------




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
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_url_with_existing_file. Retrieved 4/10 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = 'test.zip'
    var_2 = 'test'
    var_3 = True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'password_protected.zip'
    var_1 = 'password'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, password=var_1)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_level_dir.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'password_protected.zip'
    var_1 = 'wrong_password'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2, password=var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'password_protected.zip'
    var_1 = False
    var_2 = True
    var_3 = module_0.unzip(var_0, var_1, no_input=var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_handles_empty_chunks_correctly. Retrieved 5/14 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = '/tmp/test'
    var_2 = 'test.zip'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_3, var_1)



