####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no-top-dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/existing.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/existing.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_empty_zipfile_predicate. Retrieved 1/5 statements.


def test_case_0():
    var_0 = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    var_1 = [var_0]



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_chunk_filtering_in_unzip. Retrieved 5/15 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b''
    var_1 = b'valid chunk'
    var_2 = 'http://example.com/fake.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/8 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 7/8 statements.


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
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'secret'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'wrong'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no-top-level-dir-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 is not False)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 7/8 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
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
    var_2 = '.'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = 'password123'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '.'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 5/6 statements.


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
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_bad_zip_file_raises_invalid_zip_repository. Retrieved 4/6 statements.


import cookiecutter.zipfile as module_0
import locale as module_1

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.str(var_0)
    var_4 = 'Zip repository invalid.zip is not a valid zip archive:'
    var_5 = bool('Zip repository invalid.zip is not a valid zip archive:' in var_3)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = True
    var_3 = None
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = b''
    var_7 = None
    var_8 = 0
    var_9 = [call[var_8][var_8] for call in var_6]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_valid_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_local_zipfile. Retrieved 5/10 statements.
# Partially parsed test_unzip_with_empty_zipfile. Retrieved 1/7 statements.
# Partially parsed test_unzip_with_non_directory_zipfile. Retrieved 4/9 statements.
# Partially parsed test_unzip_with_password_protected_zipfile_and_valid_password. Retrieved 7/14 statements.
# Partially parsed test_unzip_with_password_protected_zipfile_and_invalid_password. Retrieved 6/13 statements.
# Partially parsed test_unzip_with_password_protected_zipfile_and_no_input. Retrieved 6/13 statements.
# Partially parsed test_unzip_with_invalid_zipfile. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 'https://invalid-url.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test/'
    var_1 = ''
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = ''
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test/'
    var_1 = ''
    var_2 = b'password'
    var_3 = False
    var_4 = 'password'
    var_5 = module_0.unzip(var_0, var_3, password=var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test/'
    var_1 = ''
    var_2 = b'password'
    var_3 = False
    var_4 = 'wrongpassword'
    var_5 = module_0.unzip(var_0, var_3, password=var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test/'
    var_1 = ''
    var_2 = b'password'
    var_3 = False
    var_4 = True
    var_5 = module_0.unzip(var_0, var_3, no_input=var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'not a zip file'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 3/6 statements.
# Partially parsed test_unzip_with_url_and_existing_zip. Retrieved 4/11 statements.
# Partially parsed test_unzip_with_local_zip. Retrieved 6/10 statements.
# Partially parsed test_unzip_with_empty_zip. Retrieved 2/8 statements.
# Partially parsed test_unzip_with_invalid_zip. Retrieved 4/9 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_password. Retrieved 7/14 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_no_input. Retrieved 6/13 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_invalid_password. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = True

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = 'repo.zip'
    var_3 = b'dummy content'

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'local_repo.zip'
    var_1 = False
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = module_0.unzip(var_2, var_1)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'empty_repo.zip'
    var_1 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid_repo.zip'
    var_1 = False
    var_2 = b'invalid zip content'
    var_3 = module_0.unzip(var_2, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected_repo.zip'
    var_1 = False
    var_2 = 'test_password'
    var_3 = 'test_dir/'
    var_4 = ''
    var_5 = module_0.unzip(var_3, var_1, password=var_2)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected_repo.zip'
    var_1 = False
    var_2 = True
    var_3 = 'test_dir/'
    var_4 = ''
    var_5 = module_0.unzip(var_3, var_1, no_input=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected_repo.zip'
    var_1 = False
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = module_0.unzip(var_2, var_1)



# Parsed testcases at query #7
#--------------------------




import zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = module_0.ZipFile(var_0)
    var_2 = var_1.namelist()
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 == var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 4/5 statements.
# Partially parsed test_unzip_with_url_and_existing_zip. Retrieved 8/12 statements.
# Partially parsed test_unzip_with_local_zip. Retrieved 6/9 statements.
# Partially parsed test_unzip_with_empty_zip. Retrieved 5/9 statements.
# Partially parsed test_unzip_with_invalid_zip. Retrieved 5/10 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 6/10 statements.
# Partially parsed test_unzip_with_invalid_password. Retrieved 5/9 statements.
# Partially parsed test_unzip_with_no_input_and_password_protected_zip. Retrieved 6/9 statements.


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
    var_0 = 'empty_repo.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'dummy content'
    var_3 = False
    var_4 = module_1.unzip(var_1, var_3)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'invalid_repo.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'dummy content'
    var_3 = False
    var_4 = module_1.unzip(var_1, var_3)

import posixpath as module_0
import cookiecutter.zipfile as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'protected_repo.zip'
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
    var_0 = 'protected_repo.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'dummy content'
    var_3 = False
    var_4 = module_1.unzip(var_1, var_3)

import posixpath as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'protected_repo.zip'
    var_1 = module_0.abspath(var_0)
    var_2 = b'dummy content'
    var_3 = False
    var_4 = True
    var_5 = module_1.unzip(var_1, var_3, no_input=var_4)



# Parsed testcases at query #9
#--------------------------




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
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)



# Parsed testcases at query #10
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
    var_0 = 'https://example.com/bad.zip'
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_uses_context_manager. Retrieved 4/9 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test/'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)



# Parsed testcases at query #12
#--------------------------




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
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'secret'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_level_dir.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/bad.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
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
    var_2 = '/tmp/test'
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
    var_2 = '/tmp/test'
    var_3 = 'test_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'wrong_password'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_empty_zipfile_predicate.




# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #18
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/invalid.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/empty.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'correct_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = 'wrong_password'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

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
    var_0 = 'http://example.com/repo.zip'
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
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    assert var_5 is False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = False
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)



