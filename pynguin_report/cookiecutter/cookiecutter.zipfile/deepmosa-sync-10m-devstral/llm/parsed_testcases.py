####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_and_existing_file_and_no_input. Retrieved 10/13 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 12/15 statements.
# Partially parsed test_unzip_with_invalid_zip_structure. Retrieved 5/8 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_valid_password. Retrieved 14/18 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_invalid_password. Retrieved 9/14 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_no_input. Retrieved 9/14 statements.
# Partially parsed test_unzip_with_invalid_zip_file. Retrieved 4/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = module_1.isdir(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '/tmp/repo.zip'
    var_1 = module_0.dirname(var_0)
    var_2 = True
    var_3 = b'dummy content'
    var_4 = 'https://example.com/repo.zip'
    var_5 = '/tmp'
    var_6 = module_1.unzip(var_4, var_2, var_5, var_2)
    var_7 = module_2.exists(var_0)
    var_8 = bool(not var_7)
    assert var_8 is True
    var_9 = module_2.exists(var_6)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = module_2.isdir(var_6)
    var_12 = bool(var_11)
    assert var_12 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = '/tmp/local_repo.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5)
    var_7 = module_1.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = module_1.isdir(var_6)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = 'file.txt'
    var_12 = [var_11]
    var_13 = module_2.join(var_6, *var_12)
    var_14 = module_1.exists(var_13)
    var_15 = bool(var_14)
    assert var_15 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/empty_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/invalid_repo.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = '/tmp/protected_repo.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = b'secret'
    var_6 = False
    var_7 = 'secret'
    var_8 = module_0.unzip(var_0, var_6, password=var_7)
    var_9 = module_1.exists(var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = module_1.isdir(var_8)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'file.txt'
    var_14 = [var_13]
    var_15 = module_2.join(var_8, *var_14)
    var_16 = module_1.exists(var_15)
    var_17 = bool(var_16)
    assert var_17 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/protected_repo.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = b'secret'
    var_6 = False
    var_7 = 'wrong'
    var_8 = module_0.unzip(var_0, var_6, password=var_7)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/protected_repo.zip'
    var_1 = 'test_dir/'
    var_2 = ''
    var_3 = 'test_dir/file.txt'
    var_4 = 'content'
    var_5 = b'secret'
    var_6 = False
    var_7 = True
    var_8 = module_0.unzip(var_0, var_6, no_input=var_7)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/invalid.zip'
    var_1 = b'not a zip file'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/bad.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_54_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_empty_zipfile_raises_invalid_zip_repository. Retrieved 7/13 statements.


import zipfile as module_0
import locale as module_1
import cookiecutter.zipfile as module_2

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = module_0.Path(var_0)
    var_2 = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_3 = module_1.str(var_1)
    var_4 = False
    var_5 = module_2.unzip(var_3, var_4)
    var_6 = module_1.str(var_1)
    var_7 = 'is empty'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_zipfile_context_manager_is_used. Retrieved 4/9 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test/'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'test_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_with_url_and_existing_file_and_no_input. Retrieved 8/10 statements.
# Partially parsed test_unzip_with_url_and_existing_file_and_user_input. Retrieved 9/11 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/7 statements.
# Partially parsed test_unzip_with_no_top_level_directory. Retrieved 5/8 statements.
# Partially parsed test_unzip_with_password_protected_and_valid_password. Retrieved 7/10 statements.
# Partially parsed test_unzip_with_password_protected_and_invalid_password. Retrieved 6/10 statements.
# Partially parsed test_unzip_with_password_protected_and_no_input. Retrieved 6/10 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/repo.zip'
    var_1 = b'dummy content'
    var_2 = 'https://example.com/repo.zip'
    var_3 = True
    var_4 = '/tmp'
    var_5 = module_0.unzip(var_2, var_3, var_4, var_3)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = module_1.exists(var_0)
    var_9 = bool(not var_8)
    assert var_9 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/repo.zip'
    var_1 = b'dummy content'
    var_2 = 'https://example.com/repo.zip'
    var_3 = True
    var_4 = '/tmp'
    var_5 = False
    var_6 = module_0.unzip(var_2, var_3, var_4, var_5)
    var_7 = module_1.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = module_1.exists(var_0)
    var_10 = bool(not var_9)
    assert var_10 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/local_repo.zip'
    var_1 = b'dummy content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/no_top_dir.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/protected.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = 'valid_password'
    var_5 = module_0.unzip(var_0, var_3, password=var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/protected.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = 'invalid_password'
    var_5 = module_0.unzip(var_0, var_3, password=var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/tmp/protected.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = True
    var_5 = module_0.unzip(var_0, var_3, no_input=var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/data/test_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/data/test_protected_repo.zip'
    var_1 = False
    var_2 = 'test_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_protected_repo.zip'
    var_1 = False
    var_2 = 'invalid_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_invalid_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_empty_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_no_top_level_dir_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_zipfile_context_manager_always_closes. Retrieved 6/8 statements.


import zipfile as module_0

def test_case_0():
    var_0 = 1
    var_1 = '.zip'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = module_0.Path(var_2)
    var_4 = 'test.txt'
    var_5 = 'test content'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_empty_zipfile_predicate.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/13 statements.
# Partially parsed test_unzip_with_empty_zip. Retrieved 1/7 statements.
# Partially parsed test_unzip_with_no_top_level_dir. Retrieved 4/9 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 9/16 statements.
# Partially parsed test_unzip_with_invalid_password. Retrieved 8/15 statements.
# Partially parsed test_unzip_with_invalid_zip_file. Retrieved 3/8 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test_dir/'
    var_1 = ''
    var_2 = 'test_dir/file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

def test_case_0():
    var_0 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test_dir/'
    var_1 = ''
    var_2 = 'test_dir/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = False
    var_6 = 'password'
    var_7 = module_0.unzip(var_0, var_5, password=var_6)
    var_8 = bool(var_2)
    assert var_8 is True
    var_9 = module_1.exists(var_7)
    var_10 = bool(var_9)
    assert var_10 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test_dir/'
    var_1 = ''
    var_2 = 'test_dir/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = False
    var_6 = 'wrong_password'
    var_7 = module_0.unzip(var_0, var_5, password=var_6)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'not a zip file'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_empty_zipfile_predicate.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_no_input_and_existing_file. Retrieved 4/5 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no-top-dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_invalid_zip_archive. Retrieved 7/9 statements.


import cookiecutter.zipfile as module_0
import locale as module_1

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.str(var_5)
    var_7 = bool(var_6 == f'Zip repository {var_0} is not a valid zip archive:')
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_url_and_existing_file. Retrieved 8/13 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/9 statements.
# Partially parsed test_unzip_with_no_top_level_directory. Retrieved 6/9 statements.
# Partially parsed test_unzip_with_password_protected_and_no_input. Retrieved 7/10 statements.
# Partially parsed test_unzip_with_invalid_password. Retrieved 7/10 statements.
# Partially parsed test_unzip_with_valid_password. Retrieved 8/11 statements.
# Partially parsed test_unzip_with_bad_zip_file. Retrieved 5/8 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = b'dummy content'
    var_5 = 'https://example.com/repo.zip'
    var_6 = True
    var_7 = module_1.unzip(var_5, var_6)
    var_8 = module_2.exists(var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'local_repo.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'dummy content'
    var_3 = False
    var_4 = module_1.unzip(var_1, var_3)
    var_5 = module_2.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = False
    var_3 = module_1.unzip(var_1, var_2)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'no_dir.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = 'file.txt'
    var_3 = b'content'
    var_4 = False
    var_5 = module_1.unzip(var_1, var_4)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = 'file.txt'
    var_3 = b'content'
    var_4 = False
    var_5 = True
    var_6 = module_1.unzip(var_1, var_4, no_input=var_5)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = 'file.txt'
    var_3 = b'content'
    var_4 = False
    var_5 = 'wrong'
    var_6 = module_1.unzip(var_1, var_4, password=var_5)

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = 'file.txt'
    var_3 = b'content'
    var_4 = False
    var_5 = 'correct'
    var_6 = module_1.unzip(var_1, var_4, password=var_5)
    var_7 = module_2.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'bad.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'not a zip file'
    var_3 = False
    var_4 = module_1.unzip(var_1, var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 5/6 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no-top-dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_no_input_and_existing_file. Retrieved 5/6 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no-dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_predicate_false. Retrieved 9/10 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = True
    var_6 = False
    var_7 = lambda _, no_input: var_6
    var_8 = module_0.prompt_and_delete(var_0, var_3)
    var_9 = bool(not var_8)
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_true. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'http://example.com/test.zip'
    var_1 = True
    var_2 = True
    var_3 = None
    var_4 = 1
    var_5 = '/'
    var_6 = var_0.rsplit(var_5, var_4)[var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 7/8 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/8 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 7/8 statements.


import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.Path(var_4)
    var_6 = var_5.exists()
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.Path(var_4)
    var_6 = var_5.exists()
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.Path(var_4)
    var_6 = var_5.exists()
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_url_and_existing_file_no_input. Retrieved 9/12 statements.
# Partially parsed test_unzip_with_url_and_existing_file_with_input. Retrieved 9/12 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/10 statements.
# Partially parsed test_unzip_with_password_protected_and_valid_password. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = b'dummy content'
    var_5 = 'https://example.com/repo.zip'
    var_6 = True
    var_7 = module_1.unzip(var_5, var_6, var_4, var_6)
    var_8 = module_2.exists(var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = module_2.exists(var_3)
    var_11 = bool(not var_10)
    assert var_11 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = b'dummy content'
    var_5 = 'https://example.com/repo.zip'
    var_6 = True
    var_7 = False
    var_8 = module_1.unzip(var_5, var_6, var_4, var_7)
    var_9 = module_2.exists(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'local_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = b'dummy content'
    var_5 = False
    var_6 = module_1.unzip(var_3, var_5)
    var_7 = module_2.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = 'valid_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = 'invalid_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_zipfile_context_manager_always_closes. Retrieved 2/4 statements.


import zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = module_0.Path(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 5/6 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_url_and_existing_file_and_no_input. Retrieved 9/13 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/14 statements.
# Partially parsed test_unzip_with_no_top_level_directory. Retrieved 7/10 statements.
# Partially parsed test_unzip_with_password_protected_file_and_valid_password. Retrieved 11/15 statements.
# Partially parsed test_unzip_with_password_protected_file_and_invalid_password. Retrieved 10/14 statements.
# Partially parsed test_unzip_with_password_protected_file_and_no_input. Retrieved 10/14 statements.
# Partially parsed test_unzip_with_invalid_zip_file. Retrieved 6/9 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = True
    var_5 = 'dummy content'
    var_6 = 'https://example.com/repo.zip'
    var_7 = module_1.unzip(var_6, var_4, var_5, var_4)
    var_8 = module_2.exists(var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = module_2.exists(var_3)
    var_11 = bool(not var_10)
    assert var_11 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'local_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'test_dir/'
    var_5 = ''
    var_6 = 'test_dir/test_file.txt'
    var_7 = 'test content'
    var_8 = False
    var_9 = module_1.unzip(var_3, var_8)
    var_10 = bool(var_6)
    assert var_10 is True
    var_11 = module_2.exists(var_9)
    var_12 = bool(var_11)
    assert var_12 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = '.'
    var_1 = 'empty_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = False
    var_5 = module_1.unzip(var_3, var_4)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = '.'
    var_1 = 'no_top_level_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'test_file.txt'
    var_5 = 'test content'
    var_6 = False
    var_7 = module_1.unzip(var_3, var_6)

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = '.'
    var_1 = 'protected_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'test_dir/'
    var_5 = ''
    var_6 = 'test_dir/test_file.txt'
    var_7 = 'test content'
    var_8 = False
    var_9 = 'valid_password'
    var_10 = module_1.unzip(var_3, var_8, password=var_9)
    var_11 = bool(var_7)
    assert var_11 is True
    var_12 = module_2.exists(var_10)
    var_13 = bool(var_12)
    assert var_13 is True

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = '.'
    var_1 = 'protected_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'test_dir/'
    var_5 = ''
    var_6 = 'test_dir/test_file.txt'
    var_7 = 'test content'
    var_8 = False
    var_9 = 'invalid_password'
    var_10 = module_1.unzip(var_3, var_8, password=var_9)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = '.'
    var_1 = 'protected_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'test_dir/'
    var_5 = ''
    var_6 = 'test_dir/test_file.txt'
    var_7 = 'test content'
    var_8 = False
    var_9 = True
    var_10 = module_1.unzip(var_3, var_8, no_input=var_9)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = '.'
    var_1 = 'invalid_repo.zip'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = 'invalid zip content'
    var_5 = False
    var_6 = module_1.unzip(var_3, var_5)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_opens_zipfile_with_context_manager. Retrieved 4/7 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'valid.zip'
    var_1 = 'dir/'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_empty_zip_file_raises_exception. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = 'Zip repository'
    var_3 = 'is empty'



# Parsed testcases at query #21
#--------------------------




import zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = module_0.ZipFile(var_0)
    var_2 = var_1.namelist()
    var_3 = len(var_2)
    var_4 = bool(var_3 != 0)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = b'some data'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------




import zipfile as module_0

def test_case_0():
    var_0 = 'valid.zip'
    var_1 = module_0.ZipFile(var_0)
    var_2 = var_1.fp
    var_3 = bool(var_1.fp is not None)
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #27
#--------------------------




import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.BytesIO(*var_1, **var_2)
    var_4 = module_1.ZipFile(var_3)
    var_5 = var_4.namelist()
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = True
    var_3 = None
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = b''
    var_7 = b'chunk1'
    var_8 = b'chunk2'



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unzip_with_valid_zip_file. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'https://example.com/valid.zip'
    var_1 = True
    var_2 = True
    var_3 = None
    var_4 = b'valid zip content'
    var_5 = 'valid_dir/'



