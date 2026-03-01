####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'local.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = False
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_url_and_existing_file. Retrieved 5/6 statements.
# Partially parsed test_unzip_with_url_and_non_existing_file. Retrieved 5/6 statements.
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

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_empty_zipfile_predicate. Retrieved 1/5 statements.


def test_case_0():
    var_0 = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_download_predicate_false. Retrieved 3/5 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'http://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = b'some data'
    var_1 = bool(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_url_and_existing_file_and_no_input. Retrieved 8/10 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/7 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)
    var_5 = module_1.isdir(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/repo.zip'
    var_1 = b'dummy content'
    var_2 = 'https://example.com/repo.zip'
    var_3 = True
    var_4 = '/tmp'
    var_5 = module_0.unzip(var_2, var_3, var_4, var_3)
    var_6 = module_1.exists(var_0)
    var_7 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/tmp/local_repo.zip'
    var_1 = b'dummy content'
    var_2 = False
    var_3 = module_0.unzip(var_0, var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'correct_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'wrong_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_level_dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_no_input_deletes_existing_file. Retrieved 7/11 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = module_1.isdir(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)
    var_4 = module_1.isdir(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = 'repo.zip'
    var_4 = True
    var_5 = module_0.unzip(var_0, var_1, var_2, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = module_1.isdir(var_3)

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



# Parsed testcases at query #8
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

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
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



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/7 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 7/8 statements.


import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/file.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = False
    var_4 = 'secret'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = module_1.exists(var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = False
    var_4 = 'wrong'
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = None
    var_4 = module_0.unzip(var_0, var_1, var_2, var_1, var_3)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)

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
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp/test'
    var_3 = module_0.unzip(var_0, var_1, var_2)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_empty_zipfile_raises_exception.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_empty_zipfile_predicate.




# Parsed testcases at query #5
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no_top_level_dir.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = b''



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_empty_zipfile_predicate.




# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = b'some data'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #11
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2, var_1)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected-repo.zip'
    var_1 = True
    var_2 = '/tmp'
    var_3 = 'secret'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)
    var_5 = module_1.exists(var_4)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/empty.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/no-top-dir.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = module_0.unzip(var_0, var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = True
    var_4 = module_0.unzip(var_0, var_1, var_2, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/path/to/protected.zip'
    var_1 = False
    var_2 = '/tmp'
    var_3 = 'wrong'
    var_4 = module_0.unzip(var_0, var_1, var_2, password=var_3)



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = b''



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_unzip_predicate_false.




# Parsed testcases at query #14
#--------------------------




import zipfile as module_0

def test_case_0():
    var_0 = 1
    var_1 = '.zip'
    var_2 = tempfile.mkstemp(suffix=var_1)[var_0]
    var_3 = module_0.Path(var_2)
    var_4 = 'w'
    var_5 = module_0.ZipFile(var_3, var_4)
    var_6 = 'test.txt'
    var_7 = 'test content'
    var_8 = var_5.writestr(var_6, var_7)
    var_9 = var_5.close()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_54_evaluates_to_false. Retrieved 3/6 statements.


import zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = module_0.Path(var_0)
    var_2 = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = b'some data'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_true. Retrieved 7/9 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = '.'
    var_3 = True
    var_4 = None
    var_5 = module_0.unzip(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'repo.zip'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = b''



# Parsed testcases at query #20
#--------------------------




import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = 'wrong'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1, no_input=var_1)

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



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = b''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_zipfile_context_manager_always_closes. Retrieved 6/15 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '.zip'
    var_1 = 'w'
    var_2 = 'test.txt'
    var_3 = 'test content'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4)



# Parsed testcases at query #23
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

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'local_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = module_1.exists(var_2)

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
    var_2 = 'secret'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)

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
    var_0 = 'https://example.com/no_top_dir.zip'
    var_1 = True
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_url_and_no_input. Retrieved 3/8 statements.


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
    var_4 = module_1.isdir(var_2)

import cookiecutter.zipfile as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'tests/data/test_protected_repo.zip'
    var_1 = False
    var_2 = 'test_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)
    var_4 = module_1.exists(var_3)
    var_5 = module_1.isdir(var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_protected_repo.zip'
    var_1 = False
    var_2 = 'wrong_password'
    var_3 = module_0.unzip(var_0, var_1, password=var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_empty_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_invalid_repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'tests/data/test_no_top_level_dir.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    var_0 = None



