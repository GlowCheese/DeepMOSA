####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 9/30 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/25 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/18 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/20 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 5/18 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = 'project_name'
    var_9 = bool(var_3)
    assert var_9 is True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = True
    var_8 = 'project_name'
    var_9 = bool(var_3)
    assert var_9 is True
    var_10 = bool(var_4)
    assert var_10 is True

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'notoplevel.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zipfile'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = 'testpass'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.'
    var_1 = 'zip_cache'
    var_2 = 'test.zip'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'http://example.com/test.zip'
    var_8 = True
    var_9 = False
    var_10 = 'test_project'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_chunk_filtering. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'data1'
    var_2 = b''
    var_3 = b'data2'
    var_4 = None
    var_5 = b'data3'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'http://example.com/test.zip'
    var_8 = 'test.zip'
    var_9 = 'project-dir/'
    var_10 = 'project-dir/file.txt'
    assert var_10 == 3
    var_11 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 16/27 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'cookiecutter.zipfile.prompt_and_delete'
    var_3 = False
    var_4 = 'return_value'
    var_5 = {var_4: var_3}
    var_6 = module_0.patch(var_2, **var_5)
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'test_dir/'
    var_11 = ''
    var_12 = 'test_dir/file.txt'
    var_13 = 'content'
    var_14 = 'cookiecutter.zipfile.requests.get'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'http://example.com/test.zip'
    var_18 = True
    var_19 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_invalid_zip_file_raises_invalid_zip_repository. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'http://example.com/invalid.zip'
    var_2 = 'Not a zip file'
    var_3 = [var_2]
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'is not a valid zip archive'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_predicate_line_31_evaluates_to_true. Retrieved 17/31 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.prompt_and_delete'
    var_7 = True
    var_8 = 'return_value'
    var_9 = {var_8: var_7}
    var_10 = module_0.patch(var_6, **var_9)
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = {}
    var_13 = module_0.patch(var_11, **var_12)
    var_14 = 'cookiecutter.zipfile.ZipFile'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'project_name/'
    var_18 = 'project_name/file.txt'
    var_19 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_20 = 'temp'
    var_21 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_bad_zipfile_exception_handling. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'not a valid zip archive'



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True
    var_3 = bool(var_1)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_line_41_predicate_false. Retrieved 7/28 statements.


def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False."
    var_1 = b''
    var_2 = None
    var_3 = [var_1, var_2, var_1]
    var_4 = 'project_dir/'
    var_5 = 'http://example.com/test.zip'
    var_6 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/21 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 7/15 statements.
# Partially parsed test_unzip_with_url_no_existing_file. Retrieved 15/30 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 16/31 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 10/20 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 10/18 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project/file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'empty'
    var_9 = bool('empty' in str(e).lower())
    assert var_9 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when zipfile lacks top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'top-level directory'
    var_11 = bool('top-level directory' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'not a valid zip archive'
    var_10 = bool('not a valid zip archive' in str(e).lower())
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist locally."
    var_1 = 'test.zip'
    var_2 = 'project/file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = 'os.path.exists'
    var_9 = False
    var_10 = 'return_value'
    var_11 = {var_10: var_9}
    var_12 = module_0.patch(var_8, **var_11)
    var_13 = 'rb'
    var_14 = 'requests.get'
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'project/file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = 'os.path.exists'
    var_9 = True
    var_10 = 'return_value'
    var_11 = {var_10: var_9}
    var_12 = module_0.patch(var_8, **var_11)
    var_13 = 'cookiecutter.zipfile.prompt_and_delete'
    var_14 = 'return_value'
    var_15 = {var_14: var_9}
    var_16 = module_0.patch(var_13, **var_15)
    var_17 = 'rb'
    var_18 = 'requests.get'
    var_19 = 'http://example.com/test.zip'
    var_20 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = False
    var_10 = 'project'

import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'correctpass'
    var_3 = 'utf-8'
    var_4 = module_0.encode(var_3)
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_1.patch(var_8, **var_9)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 6/15 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = False
    var_6 = module_1.unzip(var_0, var_5, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/20 statements.
# Partially parsed test_unzip_with_password_protected_zip_correct_password. Retrieved 9/19 statements.
# Partially parsed test_unzip_with_password_protected_zip_wrong_password. Retrieved 9/20 statements.
# Partially parsed test_unzip_no_input_with_protected_zip_no_password. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level'
    var_8 = bool('top-level' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip with invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'valid zip'
    var_7 = bool('valid zip' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'wrong'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool('password' in str(e).lower() or True)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test unzip with no_input=True and protected zip without password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool('password' in str(e).lower() or 'unlock' in str(e).lower() or True)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/21 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 4/12 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 6/14 statements.
# Partially parsed test_unzip_invalid_zipfile. Retrieved 5/11 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/21 statements.
# Partially parsed test_unzip_with_password_protected_zipfile_no_input. Retrieved 9/19 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 9/20 statements.
# Partially parsed test_unzip_returns_unzip_path. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level'
    var_8 = bool('top-level' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip with an invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a valid zipfile'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'
    var_7 = bool('not a valid zip archive' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'new_dir'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and no_input raises InvalidZipRepository.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'test_password'
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = 'test'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip returns the correct unzip_path.'
    var_1 = 'test.zip'
    var_2 = 'my_project/'
    var_3 = ''
    var_4 = 'my_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = 'my_project'
    var_9 = 'my_project'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 16/43 statements.


import locale as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = []
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = []
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = 'test_project/'
    var_8 = None
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = False
    var_13 = len(var_3)
    assert var_13 == 1
    var_14 = var_3[var_12][var_12]
    var_15 = module_0.str(var_14)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_with_url_and_new_file. Retrieved 11/31 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_with_password_provided. Retrieved 10/24 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 3/10 statements.
# Partially parsed test_unzip_with_url_existing_file_no_delete. Retrieved 4/10 statements.


def test_case_0():
    var_0 = "Test unzip with a URL when file doesn't exist yet."
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = b'test_chunk'
    var_5 = [var_4]
    var_6 = 'project-name/'
    var_7 = 'project-name/file.txt'
    var_8 = True
    var_9 = 'temp'
    var_10 = 'project-name'

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'local.zip'
    var_2 = 'project-name/'
    var_3 = 'project-name/file.txt'
    var_4 = False
    var_5 = True
    var_6 = 'temp'
    var_7 = 'project-name'

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'https://example.com/empty.zip'
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'Test that unzip raises error when zip has no top-level directory.'
    var_1 = 'https://example.com/bad.zip'
    var_2 = 'file.txt'
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip with password provided when zip is protected.'
    var_1 = 'https://example.com/protected.zip'
    var_2 = 'testpass'
    var_3 = 'project-name/'
    var_4 = 'project-name/file.txt'
    var_5 = 'Bad password'
    var_6 = [var_5]
    var_7 = None
    var_8 = True
    var_9 = 'temp'
    var_10 = 'project-name'

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'https://example.com/invalid.zip'
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not a valid zip archive'

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and user chooses not to delete.'
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_download_predicate_false_when_no_input_true. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'test.zip'
    var_3 = b'dummy content'
    var_4 = 'project/'
    var_5 = True
    var_6 = False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 13/27 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zipfile. Retrieved 8/16 statements.
# Partially parsed test_unzip_with_url_no_existing_file. Retrieved 13/26 statements.
# Partially parsed test_unzip_with_url_existing_file_force_delete. Retrieved 16/32 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 16/26 statements.
# Partially parsed test_unzip_password_protected_invalid_password. Retrieved 5/12 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = True
    var_13 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = True
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'empty'
    var_10 = bool('empty' in str(e).lower())
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when zipfile has no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'top-level directory'
    var_12 = bool('top-level directory' in str(e).lower())
    assert var_12 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'not a valid zip archive'
    var_11 = bool('not a valid zip archive' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when no cached file exists.'
    var_1 = 'https://example.com/project.zip'
    var_2 = 'clone'
    var_3 = 'temp.zip'
    var_4 = 'project/'
    var_5 = ''
    var_6 = 'project/file.txt'
    var_7 = 'content'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = True
    var_13 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when cached file exists and no_input=True.'
    var_1 = 'https://example.com/project.zip'
    var_2 = 'clone'
    var_3 = 'project.zip'
    var_4 = 'old/'
    var_5 = ''
    var_6 = 'temp.zip'
    var_7 = 'project/'
    var_8 = ''
    var_9 = 'project/file.txt'
    var_10 = 'content'
    var_11 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_12 = {}
    var_13 = module_0.patch(var_11, **var_12)
    var_14 = 'cookiecutter.zipfile.requests.get'
    var_15 = True
    var_16 = 'project'

import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and password provided.'
    var_1 = 'protected.zip'
    var_2 = 'test123'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_11 = {}
    var_12 = module_1.patch(var_10, **var_11)
    var_13 = 'cookiecutter.zipfile.ZipFile.extractall'
    var_14 = {}
    var_15 = module_1.patch(var_13, **var_14)
    var_16 = False
    var_17 = True

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zipfile with invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'clone'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 11/26 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/17 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 8/21 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 4/12 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = [var_0, var_1]
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = 'http://example.com/repo.zip'
    var_6 = True
    var_7 = '/tmp/clone'
    var_8 = module_0.unzip(var_5, var_6, var_7)
    assert var_8 == '/tmp/unzip_base/project_name'
    var_9 = 100
    var_10 = '/tmp/unzip_base'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = '/local/repo.zip'
    var_3 = False
    var_4 = '/tmp/clone'
    var_5 = module_0.unzip(var_2, var_3, var_4)
    assert var_5 == '/tmp/unzip_base/project_name'
    var_6 = '/tmp/unzip_base'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'chunk'
    var_1 = [var_0]
    var_2 = 'http://example.com/repo.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'empty'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'chunk'
    var_1 = [var_0]
    var_2 = 'file.txt'
    var_3 = 'http://example.com/repo.zip'
    var_4 = True
    var_5 = module_0.unzip(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'chunk'
    var_1 = [var_0]
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = 'http://example.com/repo.zip'
    var_5 = True
    var_6 = 'secret'
    var_7 = module_0.unzip(var_4, var_5, password=var_6)
    assert var_7 == '/tmp/unzip_base/project_name'

def test_case_0():
    var_0 = b'chunk'
    var_1 = [var_0]
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/18 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/12 statements.
# Partially parsed test_unzip_with_url_and_no_input. Retrieved 11/22 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 10/23 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 9/19 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 10/21 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_top_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist and no_input=True."
    var_1 = 'downloaded.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = [var_4]
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'http://example.com/project.zip'
    var_10 = True
    var_11 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'clone'
    var_2 = True
    var_3 = 'project.zip'
    var_4 = 'project_dir/'
    var_5 = ''
    var_6 = 'project_dir/file.txt'
    var_7 = 'content'
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'http://example.com/project.zip'
    var_10 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip with password-protected file and password provided.'
    var_1 = 'protected.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test_password'

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected file with no_input=True.'
    var_1 = 'protected.zip'
    var_2 = b'password'
    var_3 = 'project_dir/'
    var_4 = ''
    var_5 = 'project_dir/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = 'project_dir'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 11/22 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/21 statements.
# Partially parsed test_unzip_with_invalid_password_raises_error. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = b'password'
    var_5 = 'project_dir/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'password'
    var_12 = bool('password' in str(e).lower())
    assert var_12 is True

def test_case_0():
    var_0 = 'Test unzip successfully extracts password-protected zip with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test'
    var_9 = 'project_dir'

def test_case_0():
    var_0 = 'Test unzip with invalid password raises InvalidZipRepository.'
    var_1 = 'protected.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'wrong'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'password'
    var_11 = bool('password' in str(e).lower())
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file_valid. Retrieved 7/20 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 3/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/15 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/22 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'Test unzipping a local valid zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = bool(var_4)
    assert var_7 is True
    var_8 = bool(var_5)
    assert var_8 is True

def test_case_0():
    var_0 = 'Test that unzipping an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test that zipfile without top-level directory raises InvalidZipRepository.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'Test that an invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'nested'
    var_3 = 'test.zip'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = bool(var_7)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that unzip expands ~ in clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = '~/test'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 9/32 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_valid_password. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_password_protected_zip_and_invalid_password. Retrieved 10/22 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = 'requests.get'
    var_7 = 'http://example.com/test.zip'
    var_8 = True
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = True
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = b'password'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'password'
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = b'password'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'wrong'
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'nonexistent'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 16/29 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This happens when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing version instead of re-downloading.\n    '
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = False
    var_5 = 'return_value'
    var_6 = {var_5: var_4}
    var_7 = module_0.patch(var_3, **var_6)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.ZipFile'
    var_12 = {}
    var_13 = module_0.patch(var_11, **var_12)
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'temp'
    var_18 = 'project_name/'
    var_19 = True
    var_20 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_valid_zip_url. Retrieved 12/37 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/11 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 9/19 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 8/20 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test unzip with a valid zip URL.'
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'test_project/'
    var_8 = ''
    var_9 = 'clone'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with a local zip file.'
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'test.zip'
    var_5 = 'test_project/'
    var_6 = ''
    var_7 = 'file.txt'
    var_8 = 'test_project/file.txt'
    var_9 = False
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zip file.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_top_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'top-level directory'
    var_7 = bool('top-level directory' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 8
    var_7 = False
    var_8 = 'test'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = False



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False."
    var_1 = None
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_handles_bad_zip_file. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when BadZipFile exception occurs.'
    var_1 = 'http://example.com/archive.zip'
    var_2 = 'Bad zip file'
    var_3 = [var_2]
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'not a valid zip archive'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/22 statements.
# Partially parsed test_unzip_with_url_and_no_input. Retrieved 11/26 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 11/23 statements.
# Partially parsed test_unzip_password_protected_with_wrong_password_raises_error. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = 'project_name'

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist and no_input=True."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = [var_5]
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'password123'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'password123'
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'wrongpassword'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip expands user home directory in clone_to_dir.'
    var_1 = 'test.zip'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 11/24 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 7/15 statements.
# Partially parsed test_unzip_with_url_and_no_input. Retrieved 15/30 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 12/25 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 19/33 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_topdir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip file.'
    var_1 = 'bad.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True.'
    var_1 = 'temp_zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = b'PK\x03\x04'
    var_8 = [var_7]
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = 'builtins.open'
    var_14 = 'http://example.com/project.zip'
    var_15 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = 'test'
    var_12 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'existing.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = b'PK\x03\x04'
    var_8 = [var_7]
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = 'builtins.open'
    var_14 = 'os.path.exists'
    var_15 = True
    var_16 = 'return_value'
    var_17 = {var_16: var_15}
    var_18 = module_0.patch(var_14, **var_17)
    var_19 = 'cookiecutter.zipfile.prompt_and_delete'
    var_20 = 'return_value'
    var_21 = {var_20: var_15}
    var_22 = module_0.patch(var_19, **var_21)
    var_23 = 'http://example.com/project.zip'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 8/28 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/21 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 3/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/19 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 4/17 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 10/25 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = 'test_project'

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = 'test_project'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'no_dir.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_password'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = 'clone'
    var_9 = False
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_context_manager_at_line_54. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'Test that the ZipFile context manager at line 54 is used correctly.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = None
    var_10 = 'project_name'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 24/47 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'https://example.com/test.zip'
    var_2 = 'clone'
    var_3 = True
    var_4 = '/'
    var_5 = var_1.rsplit(var_4, var_3)[var_3]
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.prompt_and_delete'
    var_10 = 'return_value'
    var_11 = {var_10: var_3}
    var_12 = module_0.patch(var_9, **var_11)
    var_13 = 'cookiecutter.zipfile.requests.get'
    var_14 = {}
    var_15 = module_0.patch(var_13, **var_14)
    var_16 = '.zip'
    var_17 = False
    var_18 = 'project_name/'
    var_19 = ''
    var_20 = 'project_name/file.txt'
    var_21 = 'content'
    var_22 = 'builtins.open'
    var_23 = 'rb'
    var_24 = 'cookiecutter.zipfile.ZipFile'
    var_25 = {}
    var_26 = module_0.patch(var_24, **var_25)
    var_27 = 'project_name/'
    var_28 = 'project_name/file.txt'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 17/59 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'test.zip'
    var_2 = []
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = 'cookiecutter.zipfile.requests.get'
    var_6 = 'cookiecutter.zipfile.ZipFile.__init__'
    var_7 = 'cookiecutter.zipfile.ZipFile.namelist'
    var_8 = 'cookiecutter.zipfile.ZipFile.extractall'
    var_9 = 'cookiecutter.zipfile.ZipFile.__enter__'
    var_10 = 'cookiecutter.zipfile.ZipFile.__exit__'
    var_11 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_12 = 'temp'
    var_13 = 'https://example.com/test.zip'
    var_14 = True
    var_15 = False
    var_16 = 'prompt_and_delete'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/22 statements.
# Partially parsed test_unzip_with_password_no_input_raises. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_invalid_password. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip without top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

import genericpath as module_0

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = module_0.exists(var_8)
    var_11 = bool(var_10)
    assert var_11 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises.'
    var_1 = 'protected.zip'
    var_2 = b'password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool('password' in str(e).lower() or 'protected' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = b'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'testpass'
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and invalid password.'
    var_1 = 'protected.zip'
    var_2 = b'correctpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'wrongpass'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Invalid password'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_predicate_line_36_false. Retrieved 14/44 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing version instead of downloading.\n    '
    var_1 = 'clone'
    var_2 = 'http://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = False
    var_5 = '.zip'
    var_6 = 'test_project/'
    var_7 = ''
    var_8 = 'test_project/file.txt'
    var_9 = 'content'
    var_10 = 'test_project/'
    var_11 = 'test_project/file.txt'
    var_12 = True
    var_13 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 7/20 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5, var_2)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is empty'



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_invalid_zip_file_raises_exception. Retrieved 6/14 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = False
    var_4 = True
    var_5 = module_0.unzip(var_0, var_3, var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'is not a valid zip archive'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_local_file_valid. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/12 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/22 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_wrong_password_raises_error. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_dir/'
    var_2 = ''
    var_3 = 'project_dir/file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = True
    var_7 = bool(var_3)
    assert var_7 is True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = 'file.txt'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'no_top_dir.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'top-level directory'
    var_7 = bool('top-level directory' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a valid zip file'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'not a valid zip archive'
    var_6 = bool('not a valid zip archive' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_dir/'
    var_2 = ''
    var_3 = 'project_dir/file.txt'
    var_4 = 'content'
    var_5 = 'new_dir'
    var_6 = 'nested'
    var_7 = False
    var_8 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'test_password'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = False
    var_9 = True
    var_10 = bool(var_4)
    assert var_10 is True
    var_11 = bool(var_5)
    assert var_11 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'project_dir/'
    var_2 = ''
    var_3 = 'project_dir/file.txt'
    var_4 = 'content'
    var_5 = 'correct_password'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = False
    var_9 = True
    var_10 = 'wrong_password'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'invalid password'
    var_13 = bool('invalid password' in str(e).lower())
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 16/37 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.'
    var_1 = 'cookiecutter_repo'
    var_2 = True
    var_3 = 'http://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = b'dummy content'
    var_6 = 'cookiecutter.zipfile.prompt_and_delete'
    var_7 = 'requests_get'
    var_8 = 0
    var_9 = {var_7: var_8}
    var_10 = 'cookiecutter.zipfile.requests.get'
    var_11 = 'cookiecutter.zipfile.sys.exit'
    var_12 = 'cookiecutter.zipfile.read_user_yes_no'
    var_13 = True
    var_14 = False
    var_15 = module_0.unzip(var_3, var_13, var_1, var_14)
    var_16 = var_9['requests_get']
    assert var_16 == 0



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 16/44 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = 'clone'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = None
    var_5 = b'chunk3'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = 'test_project/'
    var_9 = 'test_project/file.txt'
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = 'temp'
    var_12 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_13 = 'https://example.com/test.zip'
    var_14 = True
    var_15 = 'test.zip'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 10/32 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/25 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/27 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 12/38 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 6/23 statements.


def test_case_0():
    var_0 = "Test unzip with a URL when zip file doesn't exist locally."
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = 'https://example.com/test.zip'
    var_9 = True
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'local_project/'
    var_3 = ''
    var_4 = 'local_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = 'local_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = []
    var_2 = 0
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = 'https://example.com/empty.zip'
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'empty'
    var_8 = bool('empty' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zip has no top-level directory.'
    var_1 = []
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 0
    var_5 = 'cookiecutter.zipfile.requests.get'
    var_6 = 'https://example.com/bad.zip'
    var_7 = True
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'top-level directory'
    var_10 = bool('top-level directory' in str(e).lower())
    assert var_10 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'cookiecutter.zipfile.requests.get'
    var_2 = 'https://example.com/invalid.zip'
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'not a valid zip archive'
    var_6 = bool('not a valid zip archive' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and provided password.'
    var_1 = []
    var_2 = 'protected_project/'
    var_3 = ''
    var_4 = 'protected_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'protected.zip'
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'https://example.com/protected.zip'
    var_10 = True
    var_11 = 'test'
    var_12 = 'protected_project'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = []
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 12/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = True
    var_12 = False
    var_13 = 'project_dir'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/17 statements.
# Partially parsed test_unzip_url_new_file. Retrieved 10/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 5/12 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/22 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 12/24 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 8/18 statements.
# Partially parsed test_unzip_url_with_existing_file_and_delete. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist."
    var_1 = 'clone'
    var_2 = 'remote.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'https://example.com/project.zip'
    var_8 = 'requests.get'
    var_9 = True
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level'
    var_8 = bool('top-level' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'clone'
    var_2 = 'protected.zip'
    var_3 = 'test_password'
    var_4 = 'utf-8'
    var_5 = module_0.encode(var_4)
    var_6 = 'project_name/'
    var_7 = ''
    var_8 = 'project_name/file.txt'
    var_9 = 'content'
    var_10 = False
    var_11 = 'project_name'
    var_12 = bool(var_8)
    assert var_12 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and invalid password.'
    var_1 = 'clone'
    var_2 = 'protected.zip'
    var_3 = 'correct_password'
    var_4 = 'utf-8'
    var_5 = module_0.encode(var_4)
    var_6 = 'project_name/'
    var_7 = ''
    var_8 = 'project_name/file.txt'
    var_9 = 'content'
    var_10 = False
    var_11 = 'wrong_password'
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Invalid password'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'new_clone_dir'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and user chooses to delete.'
    var_1 = 'clone'
    var_2 = 'project.zip'
    var_3 = 'old_project/'
    var_4 = ''
    var_5 = 'old_project/old.txt'
    var_6 = 'old content'
    var_7 = 'new.zip'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/12 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_expanduser_path. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False

import cookiecutter.zipfile as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip expands user home directory in clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'expanduser'
    var_7 = False
    var_8 = '~/cookiecutter'
    var_9 = module_0.unzip(var_5, var_7, var_8)
    var_10 = module_1.Path(var_9)
    var_11 = var_10.exists()
    var_12 = bool(var_11)
    assert var_12 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_bad_zip_file_raises_invalid_zip_repository. Retrieved 9/19 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'not a valid zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = 'Bad zip file'
    var_8 = [var_7]
    var_9 = False
    var_10 = module_1.unzip(var_0, var_9, var_2)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'is not a valid zip archive'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception_handling. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is not a valid zip archive'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_iter_content_chunk_predicate_evaluates_to_false. Retrieved 9/31 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks.'
    var_1 = 'http://example.com/repo.zip'
    var_2 = b'data'
    var_3 = b''
    var_4 = b'more_data'
    var_5 = 'project_name/'
    var_6 = True
    var_7 = b'data'
    var_8 = b'more_data'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unzip_downloads_zipfile_when_download_is_true. Retrieved 14/35 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True and file is opened for writing.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = 'requests.get'
    var_5 = 'project/'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = 'tempfile.mkdtemp'
    var_8 = 'temp'
    var_9 = 'os.path.exists'
    var_10 = False
    var_11 = 'return_value'
    var_12 = {var_11: var_10}
    var_13 = module_0.patch(var_9, **var_12)
    var_14 = 'builtins.open'
    var_15 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unzip_context_manager_closes_zipfile. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = bool(var_5)
    assert var_9 is True
    var_10 = 'test_project'
    var_11 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'extract'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = 'test_project'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'extract'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'no_toplevel.zip'
    var_1 = 'extract'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level'
    var_8 = bool('top-level' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'extract'
    var_2 = 'This is not a valid zip file'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'valid zip'
    var_7 = bool('valid zip' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'new_extract_dir'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 8/28 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/19 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/22 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/23 statements.
# Partially parsed test_unzip_with_password_protection. Retrieved 10/31 statements.


def test_case_0():
    var_0 = b'PK\x03\x04'
    var_1 = b'chunk1'
    var_2 = b'chunk2'
    var_3 = [var_1, var_2]
    var_4 = 'project_dir/'
    var_5 = 'project_dir/file.txt'
    var_6 = 'https://example.com/project.zip'
    var_7 = True
    var_8 = 'project_dir'

def test_case_0():
    var_0 = 'myproject/'
    var_1 = 'myproject/file.txt'
    var_2 = False
    var_3 = True
    var_4 = 'myproject'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = [var_0]
    var_2 = 'https://example.com/empty.zip'
    var_3 = True
    var_4 = module_0.unzip(var_2, var_3, no_input=var_3)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = [var_0]
    var_2 = 'file.txt'
    var_3 = 'https://example.com/bad.zip'
    var_4 = True
    var_5 = module_0.unzip(var_3, var_4, no_input=var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = [var_0]
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 0
    var_5 = [var_4]
    var_6 = 'https://example.com/protected.zip'
    var_7 = True
    var_8 = 'secret'
    var_9 = module_0.unzip(var_6, var_7, no_input=var_7, password=var_8)
    var_10 = bool(var_9 is not None)
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unzip_predicate_line_54_false. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = 'project_dir'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 20/44 statements.


import locale as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = b'PK\x03\x04'
    var_7 = b'some'
    var_8 = b''
    var_9 = b'data'
    var_10 = [var_6, var_7, var_8, var_9, var_8]
    var_11 = iter(var_10)
    var_12 = 'test_dir/'
    var_13 = 'test_dir/file.txt'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = 0
    var_17 = module_0.str(var_10)
    var_18 = b'PK\x03\x04'
    var_19 = bool(b'PK\x03\x04' in var_17)
    assert var_19 is True
    var_20 = module_0.str(var_10)
    var_21 = b'some'
    var_22 = bool(b'some' in var_20)
    assert var_22 is True
    var_23 = module_0.str(var_10)
    var_24 = b'data'
    var_25 = bool(b'data' in var_23)
    assert var_25 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 15/32 statements.
# Partially parsed test_unzip_with_local_file_extracts_zipfile. Retrieved 11/22 statements.
# Partially parsed test_unzip_raises_on_empty_zipfile. Retrieved 8/19 statements.
# Partially parsed test_unzip_raises_on_missing_top_level_directory. Retrieved 10/21 statements.
# Partially parsed test_unzip_raises_on_invalid_zip_file. Retrieved 9/18 statements.
# Partially parsed test_unzip_with_password_protected_zipfile. Retrieved 13/27 statements.
# Partially parsed test_unzip_prompts_for_password_when_needed. Retrieved 14/26 statements.
# Partially parsed test_unzip_raises_on_wrong_password. Retrieved 4/9 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a zipfile from URL.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'rb'
    var_8 = 'requests.get'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = True
    var_14 = 'return_value'
    var_15 = {var_14: var_13}
    var_16 = module_0.patch(var_12, **var_15)
    var_17 = 'http://example.com/test.zip'
    var_18 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip extracts a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = True
    var_12 = 'project_name'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = True
    var_8 = module_1.unzip(var_0, var_6, var_2, var_7)
    var_9 = bool(False)
    assert var_9 is True

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when top-level is not a directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = True
    var_10 = module_1.unzip(var_2, var_8, var_4, var_9)
    var_11 = bool(False)
    assert var_11 is True

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = True
    var_9 = module_1.unzip(var_0, var_7, var_2, var_8)
    var_10 = bool(False)
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip handles password protected zipfile with provided password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'test_password'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = True
    var_13 = 'test_password'
    var_14 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip prompts user for password when zipfile is protected.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'test_password'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.read_repo_password'
    var_12 = 'test_password'
    var_13 = 'return_value'
    var_14 = {var_13: var_12}
    var_15 = module_0.patch(var_11, **var_14)
    var_16 = False
    var_17 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when wrong password is provided.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unzip_with_url_and_existing_file_no_delete. Retrieved 6/18 statements.
# Partially parsed test_unzip_with_url_and_download. Retrieved 9/29 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/23 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/15 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/17 statements.
# Partially parsed test_unzip_with_password_protected_file_with_password. Retrieved 11/26 statements.
# Partially parsed test_unzip_with_password_protected_file_no_input_raises_error. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'Test unzip with URL when file exists and user chooses not to delete.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = b'fake zip content'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist and needs download."
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'temp'
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = 'local.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'temp'
    var_7 = True
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = b'not a zip file'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and provided password.'
    var_1 = 'protected.zip'
    var_2 = b'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'temp'
    var_8 = True
    var_9 = False
    var_10 = 'testpass'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input raises error.'
    var_1 = 'protected.zip'
    var_2 = b'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'temp'
    var_8 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unzip_predicate_line_54_evaluates_to_false. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This ensures the zipfile is not empty when processing a valid archive.\n    '
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = bool(var_3)
    assert var_7 is True
    var_8 = 'project_dir'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_unzip_raises_error_when_zip_is_empty. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Zip repository'
    var_9 = 'is empty'



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_unzip_with_valid_zipfile_predicate_line_54. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = len(var_2)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 0
    var_9 = zip_file.namelist()[var_8]
    var_10 = '/'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/21 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_expanduser_path. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

import genericpath as module_0

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = module_0.exists(var_8)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'password'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool('password' in str(e).lower() or 'protected' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'testpass'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'testpass'
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip expands user paths correctly.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_predicate_line_31_evaluates_to_true. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True.'
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = 'project_name/'
    var_5 = True
    var_6 = False
    var_7 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 13/41 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 13/32 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/20 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/21 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 2/14 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with a URL when no cached file exists.'
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = [var_2]
    var_9 = 'test_project/'
    var_10 = 'test_project/file.txt'
    var_11 = 'http://example.com/test.zip'
    var_12 = True
    var_13 = module_0.unzip(var_11, var_12, var_5, var_12)
    var_14 = 'test_project'
    var_15 = bool('test_project' in var_13)
    assert var_15 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = []
    var_2 = 'local_project/'
    var_3 = ''
    var_4 = 'local_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = 'local_project/'
    var_9 = 'local_project/file.txt'
    var_10 = '/local/path/file.zip'
    var_11 = False
    var_12 = True
    var_13 = module_0.unzip(var_10, var_11, var_5, var_12)
    var_14 = 'local_project'
    var_15 = bool('local_project' in var_13)
    assert var_15 is True

def test_case_0():
    var_0 = 'Test that unzipping an empty zip file raises InvalidZipRepository.'
    var_1 = 'clone'
    var_2 = 'http://example.com/empty.zip'
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test that unzipping a file without top-level directory raises InvalidZipRepository.'
    var_1 = 'clone'
    var_2 = 'file.txt'
    var_3 = 'http://example.com/bad.zip'
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'top-level'
    var_7 = bool('top-level' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test that a bad zip file raises InvalidZipRepository.'
    var_1 = 'clone'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_url_and_no_existing_file. Retrieved 11/39 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 6/14 statements.
# Partially parsed test_unzip_with_password_provided. Retrieved 11/26 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/27 statements.


def test_case_0():
    var_0 = "Test unzip with a URL when the zip file doesn't exist locally."
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'clone'
    var_8 = 'requests.get'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'local_project'
    var_2 = 'file.txt'
    var_3 = 'local content'
    var_4 = 'local.zip'
    var_5 = 'file.txt'
    var_6 = 'local_project/file.txt'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = 'local_project'

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected_project'
    var_2 = 'file.txt'
    var_3 = 'protected content'
    var_4 = 'protected.zip'
    var_5 = 'file.txt'
    var_6 = 'protected_project/file.txt'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = 'test_password'

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'nonexistent'
    var_8 = 'clone'
    var_9 = False
    var_10 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_bad_zipfile_exception_handling. Retrieved 8/17 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = module_0.unzip(var_0, var_6, var_2)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'is not a valid zip archive'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_predicate_line_39_false. Retrieved 13/29 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to False.'
    var_1 = b''
    var_2 = None
    var_3 = 'requests.get'
    var_4 = 'test_dir/'
    var_5 = 'zipfile.ZipFile'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'builtins.open'
    var_11 = 'https://example.com/test.zip'
    var_12 = True
    var_13 = 1024



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 12/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 8/16 statements.
# Partially parsed test_unzip_with_url_no_existing_file. Retrieved 17/33 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 18/34 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 18/35 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = True
    var_12 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = True
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'empty'
    var_10 = bool('empty' in str(e).lower())
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'top-level directory'
    var_12 = bool('top-level directory' in str(e).lower())
    assert var_12 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'not a valid zip archive'
    var_11 = bool('not a valid zip archive' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist."
    var_1 = 'remote.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'os.path.exists'
    var_11 = False
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'rb'
    var_16 = 'requests.get'
    var_17 = 'http://example.com/project.zip'
    var_18 = True
    var_19 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'remote.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'os.path.exists'
    var_11 = True
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'cookiecutter.zipfile.prompt_and_delete'
    var_16 = 'return_value'
    var_17 = {var_16: var_11}
    var_18 = module_0.patch(var_15, **var_17)
    var_19 = 'rb'
    var_20 = 'requests.get'
    var_21 = 'http://example.com/project.zip'
    var_22 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = b'password'
    var_7 = 'project/'
    var_8 = ''
    var_9 = 'project/file.txt'
    var_10 = 'content'
    var_11 = 'clone'
    var_12 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_13 = {}
    var_14 = module_0.patch(var_12, **var_13)
    var_15 = False
    var_16 = True
    var_17 = 'password'
    var_18 = 'project'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_bad_zip_file_raises_invalid_zip_repository. Retrieved 6/16 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4, var_2)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'is not a valid zip archive'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 15/34 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/28 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/22 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/23 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 5/17 statements.
# Partially parsed test_unzip_with_password_protection. Retrieved 9/27 statements.


import locale as module_0

def test_case_0():
    var_0 = 'clone'
    var_1 = b'PK\x03\x04'
    var_2 = b'\x00'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = var_1 + var_4
    var_6 = [var_5]
    var_7 = 'project_name/'
    var_8 = 'project_name/file.txt'
    var_9 = 'https://example.com/project.zip'
    var_10 = True
    var_11 = 'temp'
    var_12 = 'project_name'
    var_13 = var_2 / var_12
    var_14 = module_0.str(var_13)

def test_case_0():
    var_0 = 'clone'
    var_1 = 'local.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = False
    var_6 = True
    var_7 = 'temp'
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'empty.zip'
    var_2 = b'PK\x03\x04'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'notoplevel.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'file.txt'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'bad.zip'
    var_2 = b'not a zip'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'protected.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = 'Bad password'
    var_6 = [var_5]
    var_7 = False
    var_8 = True
    var_9 = 'correct_password'

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 24/52 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'https://example.com/test.zip'
    var_2 = 'clone'
    var_3 = 1
    var_4 = '/'
    var_5 = var_1.rsplit(var_4, var_3)[var_3]
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.prompt_and_delete'
    var_10 = True
    var_11 = 'return_value'
    var_12 = {var_11: var_10}
    var_13 = module_0.patch(var_9, **var_12)
    var_14 = 'cookiecutter.zipfile.requests.get'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'cookiecutter.zipfile.ZipFile'
    var_18 = {}
    var_19 = module_0.patch(var_17, **var_18)
    var_20 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_21 = 'temp'
    var_22 = 'test_project/'
    var_23 = b'test'
    var_24 = 'builtins.open'
    var_25 = True
    var_26 = False
    var_27 = '__self__'
    var_28 = None



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for empty chunks.'
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_zipfile_context_manager_predicate. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = bool(var_5 > 0)
    assert var_9 is True
    var_10 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/18 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/12 statements.
# Partially parsed test_unzip_url_download_and_extract. Retrieved 14/27 statements.
# Partially parsed test_unzip_url_reuse_existing. Retrieved 12/23 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 9/17 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 10/21 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts from URL.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'requests.get'
    var_7 = 'cookiecutter.zipfile.prompt_and_delete'
    var_8 = True
    var_9 = 'return_value'
    var_10 = {var_9: var_8}
    var_11 = module_0.patch(var_7, **var_10)
    var_12 = 'clone'
    var_13 = 'http://example.com/test.zip'
    var_14 = 'project_name'
    var_15 = 'os'
    var_16 = __import__(var_15)

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip reuses existing cached zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.prompt_and_delete'
    var_7 = False
    var_8 = 'return_value'
    var_9 = {var_8: var_7}
    var_10 = module_0.patch(var_6, **var_9)
    var_11 = 'clone'
    var_12 = True
    var_13 = 'http://example.com/test.zip'
    var_14 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input raises error.'
    var_1 = 'protected.zip'
    var_2 = b'correct_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'password'
    var_12 = bool('password' in str(e).lower())
    assert var_12 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_creates_clone_to_dir. Retrieved 25/37 statements.
# Partially parsed test_unzip_with_existing_url_prompts_deletion. Retrieved 24/37 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 17/30 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 19/31 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 20/32 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'clone'
    var_2 = 'http://example.com/repo.zip'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = True
    var_14 = 'return_value'
    var_15 = {var_14: var_13}
    var_16 = module_0.patch(var_12, **var_15)
    var_17 = b'chunk1'
    var_18 = b'chunk2'
    var_19 = 'project/'
    var_20 = 'project/file.txt'
    var_21 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_22 = 'temp'
    var_23 = 'cookiecutter.zipfile.os.path.exists'
    var_24 = False
    var_25 = 'return_value'
    var_26 = {var_25: var_24}
    var_27 = module_0.patch(var_23, **var_26)
    var_28 = 'cookiecutter.zipfile.os.path.join'
    var_29 = '/'
    var_30 = lambda *args: var_29.join(args)
    var_31 = 'side_effect'
    var_32 = {var_31: var_30}
    var_33 = module_0.patch(var_28, **var_32)

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip prompts for deletion when zip already exists.'
    var_1 = 'http://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.os.path.exists'
    var_7 = True
    var_8 = 'return_value'
    var_9 = {var_8: var_7}
    var_10 = module_0.patch(var_6, **var_9)
    var_11 = 'cookiecutter.zipfile.prompt_and_delete'
    var_12 = 'return_value'
    var_13 = {var_12: var_7}
    var_14 = module_0.patch(var_11, **var_13)
    var_15 = 'cookiecutter.zipfile.requests.get'
    var_16 = {}
    var_17 = module_0.patch(var_15, **var_16)
    var_18 = 'cookiecutter.zipfile.ZipFile'
    var_19 = {}
    var_20 = module_0.patch(var_18, **var_19)
    var_21 = b'content'
    var_22 = 'project/'
    var_23 = 'project/file.txt'
    var_24 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_25 = 'temp'
    var_26 = 'cookiecutter.zipfile.os.path.join'
    var_27 = '/'
    var_28 = lambda *args: var_27.join(args)
    var_29 = 'side_effect'
    var_30 = {var_29: var_28}
    var_31 = module_0.patch(var_26, **var_30)
    var_32 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip works with local file path.'
    var_1 = 'repo.zip'
    var_2 = 'fake zip content'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.os.path.abspath'
    var_10 = 'project/'
    var_11 = 'project/file.txt'
    var_12 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_13 = 'temp'
    var_14 = 'cookiecutter.zipfile.os.path.join'
    var_15 = '/'
    var_16 = lambda *args: var_15.join(args)
    var_17 = 'side_effect'
    var_18 = {var_17: var_16}
    var_19 = module_0.patch(var_14, **var_18)
    var_20 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'http://example.com/repo.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = 'cookiecutter.zipfile.os.path.exists'
    var_6 = False
    var_7 = 'return_value'
    var_8 = {var_7: var_6}
    var_9 = module_0.patch(var_5, **var_8)
    var_10 = 'cookiecutter.zipfile.requests.get'
    var_11 = {}
    var_12 = module_0.patch(var_10, **var_11)
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = {}
    var_15 = module_0.patch(var_13, **var_14)
    var_16 = b'content'
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = 'temp'
    var_19 = 'cookiecutter.zipfile.os.path.join'
    var_20 = '/'
    var_21 = lambda *args: var_20.join(args)
    var_22 = 'side_effect'
    var_23 = {var_22: var_21}
    var_24 = module_0.patch(var_19, **var_23)
    var_25 = True
    var_26 = bool(False)
    assert var_26 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'http://example.com/repo.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = 'cookiecutter.zipfile.os.path.exists'
    var_6 = False
    var_7 = 'return_value'
    var_8 = {var_7: var_6}
    var_9 = module_0.patch(var_5, **var_8)
    var_10 = 'cookiecutter.zipfile.requests.get'
    var_11 = {}
    var_12 = module_0.patch(var_10, **var_11)
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = {}
    var_15 = module_0.patch(var_13, **var_14)
    var_16 = b'content'
    var_17 = 'file.txt'
    var_18 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_19 = 'temp'
    var_20 = 'cookiecutter.zipfile.os.path.join'
    var_21 = '/'
    var_22 = lambda *args: var_21.join(args)
    var_23 = 'side_effect'
    var_24 = {var_23: var_22}
    var_25 = module_0.patch(var_20, **var_24)
    var_26 = True
    var_27 = bool(False)
    assert var_27 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 11/24 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 7/16 statements.
# Partially parsed test_unzip_url_with_no_input_no_download. Retrieved 17/27 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 13/25 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 12/25 statements.
# Partially parsed test_unzip_password_protected_with_user_input. Retrieved 13/22 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'empty'
    var_9 = bool('empty' in str(e).lower())
    assert var_9 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'top-level directory'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'not a valid zip archive'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'os.path.exists'
    var_11 = True
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'cookiecutter.zipfile.prompt_and_delete'
    var_16 = False
    var_17 = 'return_value'
    var_18 = {var_17: var_16}
    var_19 = module_0.patch(var_15, **var_18)
    var_20 = 'http://example.com/test.zip'
    var_21 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password provided.'
    var_1 = 'protected.zip'
    var_2 = b'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = 'test_password'
    var_13 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = b'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'password protected'
    var_15 = bool('password protected' in str(e).lower())
    assert var_15 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip prompts user for password.'
    var_1 = 'protected.zip'
    var_2 = b'user_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.read_repo_password'
    var_12 = 'user_password'
    var_13 = 'return_value'
    var_14 = {var_13: var_12}
    var_15 = module_0.patch(var_11, **var_14)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception_handling. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is not a valid zip archive'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 9/25 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/18 statements.
# Partially parsed test_unzip_empty_repository_raises_error. Retrieved 2/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 3/16 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/29 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = b'fake'
    var_2 = b'zip'
    var_3 = b'data'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'project-name/'
    var_6 = 'project-name/file.txt'
    var_7 = True
    var_8 = 'project-name'

def test_case_0():
    var_0 = '/path/to/local/repo.zip'
    var_1 = 'project-name/'
    var_2 = 'project-name/file.txt'
    var_3 = False
    var_4 = 'project-name'

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = True
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'empty'
    var_4 = bool('empty' in str(e).lower())
    assert var_4 is True

def test_case_0():
    var_0 = 'https://example.com/notoplevel.zip'
    var_1 = 'file.txt'
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'top-level'
    var_5 = bool('top-level' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = 'test_password'
    var_2 = b'fake'
    var_3 = b'zip'
    var_4 = [var_2, var_3]
    var_5 = 'project-name/'
    var_6 = 'project-name/file.txt'
    var_7 = 'Bad password'
    var_8 = [var_7]
    var_9 = None
    var_10 = True
    var_11 = 'project-name'

def test_case_0():
    var_0 = 'https://example.com/invalid.zip'
    var_1 = b'not'
    var_2 = b'a'
    var_3 = b'zip'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'not a valid zip archive'
    var_8 = bool('not a valid zip archive' in str(e).lower())
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_on_bad_zipfile. Retrieved 8/19 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = module_0.unzip(var_0, var_4, var_2, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'is not a valid zip archive'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 6/23 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/15 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/11 statements.
# Partially parsed test_unzip_without_top_level_directory_raises_error. Retrieved 4/12 statements.
# Partially parsed test_unzip_with_password_protection. Retrieved 7/18 statements.
# Partially parsed test_unzip_invalid_password_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_input_with_password_protected_raises_error. Retrieved 7/17 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = b'PK\x03\x04'
    var_1 = 'project_name/'
    var_2 = 'project_name/file.txt'
    var_3 = 'http://example.com/repo.zip'
    var_4 = True
    var_5 = module_0.unzip(var_3, var_4)
    assert var_5 == '/tmp/tmpdir/project_name'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = '/local/path/repo.zip'
    var_3 = False
    var_4 = module_0.unzip(var_2, var_3)
    assert var_4 == '/tmp/tmpdir/project_name'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = '/local/path/repo.zip'
    var_2 = False
    var_3 = module_0.unzip(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'top-level directory'
    var_6 = bool('top-level directory' in str(e).lower())
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = 'password protected'
    var_3 = [var_2]
    var_4 = '/local/path/repo.zip'
    var_5 = False
    var_6 = 'secret'
    var_7 = module_0.unzip(var_4, var_5, password=var_6)
    assert var_7 == '/tmp/tmpdir/project_name'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = 'password protected'
    var_3 = [var_2]
    var_4 = '/local/path/repo.zip'
    var_5 = False
    var_6 = 'wrong'
    var_7 = module_0.unzip(var_4, var_5, password=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'invalid password'
    var_10 = bool('invalid password' in str(e).lower())
    assert var_10 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not a valid zip archive'
    var_5 = bool('not a valid zip archive' in str(e).lower())
    assert var_5 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = 'password protected'
    var_3 = [var_2]
    var_4 = '/local/path/repo.zip'
    var_5 = False
    var_6 = True
    var_7 = module_0.unzip(var_4, var_5, no_input=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'unable to unlock'
    var_10 = bool('unable to unlock' in str(e).lower())
    assert var_10 is True

def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 13/36 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/20 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 6/12 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 11/21 statements.
# Partially parsed test_unzip_password_protected_no_password_no_input_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 9/21 statements.


import requests.cookies as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a zipfile from URL.'
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = module_0.MockResponse(var_4)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'cookiecutter.zipfile.prompt_and_delete'
    var_10 = True
    var_11 = lambda path, no_input: var_10
    var_12 = 'http://example.com/test.zip'
    var_13 = 'test_project'
    var_14 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with a local zipfile path.'
    var_1 = 'test.zip'
    var_2 = 'local_project/'
    var_3 = ''
    var_4 = 'local_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = 'local_project'
    var_9 = 'local_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'
    var_8 = bool('top-level directory' in str(e).lower())
    assert var_8 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False
    var_4 = True
    var_5 = module_0.unzip(var_0, var_3, var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'valid zip archive'
    var_8 = bool('valid zip archive' in str(e).lower())
    assert var_8 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and provided password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'protected_project/'
    var_4 = ''
    var_5 = 'protected_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = True
    var_11 = 'protected_project'

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zip with no_input=True and no password.'
    var_1 = 'protected.zip'
    var_2 = 'protected_project/'
    var_3 = ''
    var_4 = b'testpass'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'password protected'
    var_9 = bool('password protected' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'new_dir'
    var_7 = bool(not var_4)
    assert var_7 is True
    var_8 = False
    var_9 = True
    var_10 = 'project'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_on_bad_zipfile. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'clone'
    var_2 = 'fake.zip'
    var_3 = 'This is not a valid zip file'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'is not a valid zip archive'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 9/36 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'test.zip'
    var_2 = 'clone'
    var_3 = 'project/'
    var_4 = 'project/file.txt'
    var_5 = 'http://example.com/test.zip'
    var_6 = True
    var_7 = False
    var_8 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This happens when prompt_and_delete returns False, meaning the user\n    wants to reuse the existing version.\n    '
    var_1 = 'http://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'project/'
    var_4 = True
    var_5 = False



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b''
    var_2 = b'some data'
    var_3 = bool(not var_1)
    assert var_3 is True
    var_4 = bool(var_2)
    assert var_4 is True
    var_5 = b''
    var_6 = b'data1'
    var_7 = b'data2'
    var_8 = [var_5, var_6, var_5, var_7, var_5]
    var_9 = [chunk for chunk in var_8 if chunk]
    var_10 = bool(var_9 == [b'data1', b'data2'])
    assert var_10 is True
    var_11 = len(var_9)
    assert var_11 == 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception_handling. Retrieved 7/19 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = b'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = module_0.unzip(var_0, var_4, var_2, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is not a valid zip archive'



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'test data'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 20/42 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (os.path.exists(zip_path)) evaluates to True.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = True
    var_5 = 'return_value'
    var_6 = {var_5: var_4}
    var_7 = module_0.patch(var_3, **var_6)
    var_8 = b'test data'
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = 'project_name/'
    var_11 = 'project_name/file.txt'
    var_12 = None
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = 'temp'
    var_16 = 'http://example.com/test.zip'
    var_17 = 'cookiecutter.zipfile.os.path.join'
    var_18 = lambda *args: str(Path(*args))
    var_19 = 'side_effect'
    var_20 = {var_19: var_18}
    var_21 = module_0.patch(var_17, **var_20)
    var_22 = False
    var_23 = {}
    var_24 = module_0.patch(var_3, **var_23)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_on_bad_zipfile. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception is caught and converted to InvalidZipRepository.'
    var_1 = 'Bad zip file'
    var_2 = [var_1]
    var_3 = '/path/to/bad.zip'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'is not a valid zip archive'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 15/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing version instead of downloading.\n    '
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = 'cookiecutter.zipfile.prompt_and_delete'
    var_9 = False
    var_10 = 'return_value'
    var_11 = {var_10: var_9}
    var_12 = module_0.patch(var_8, **var_11)
    var_13 = 'cookiecutter.zipfile.requests.get'
    var_14 = {}
    var_15 = module_0.patch(var_13, **var_14)
    var_16 = True
    var_17 = None
    var_18 = 'project_name'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true_when_zip_path_exists. Retrieved 14/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.prompt_and_delete'
    var_8 = False
    var_9 = 'return_value'
    var_10 = {var_9: var_8}
    var_11 = module_0.patch(var_7, **var_10)
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = {}
    var_14 = module_0.patch(var_12, **var_13)
    var_15 = f'http://example.com/{var_2}'
    var_16 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_predicate_line_39_evaluates_to_false. Retrieved 13/29 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = b'some data'
    var_7 = b''
    var_8 = b'more data'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_predicate_line_39_evaluates_to_false. Retrieved 11/31 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to False.'
    var_1 = b''
    var_2 = None
    var_3 = [var_1, var_2, var_1]
    var_4 = 'test.zip'
    var_5 = 'test_project/'
    var_6 = ''
    var_7 = 'test_project/file.txt'
    var_8 = 'content'
    var_9 = 'http://example.com/test.zip'
    var_10 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_predicate_line_39_evaluates_to_true. Retrieved 22/54 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = b'test data chunk'
    var_2 = b''
    var_3 = b'another chunk'
    var_4 = 'cookiecutter.zipfile.requests.get'
    var_5 = 'test.zip'
    var_6 = 'test_dir/'
    var_7 = ''
    var_8 = 'test_dir/file.txt'
    var_9 = 'content'
    var_10 = 'test_dir/'
    var_11 = 'test_dir/file.txt'
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_14 = 'temp'
    var_15 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_16 = {}
    var_17 = module_0.patch(var_15, **var_16)
    var_18 = 'builtins.open'
    var_19 = 'http://example.com/test.zip'
    var_20 = True
    var_21 = 0
    var_22 = "The predicate 'if chunk:' should evaluate to True for non-empty chunks"



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 9/31 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 41 evaluates to True when chunk is not empty.'
    var_1 = 'https://example.com/test.zip'
    var_2 = b'test data 1'
    var_3 = b'test data 2'
    var_4 = b''
    var_5 = 'project_name/'
    var_6 = []
    var_7 = True
    var_8 = bool(var_2 in var_6)
    assert var_8 is True
    var_9 = bool(var_3 in var_6)
    assert var_9 is True
    var_10 = bool(var_4 not in var_6)
    assert var_10 is True
    var_11 = len(var_6)
    assert var_11 == 2



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 15/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when is_url is True, the zip file exists, and prompt_and_delete\n    returns False (user chooses to reuse existing version).\n    '
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = False
    var_5 = 'return_value'
    var_6 = {var_5: var_4}
    var_7 = module_0.patch(var_3, **var_6)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.ZipFile'
    var_12 = {}
    var_13 = module_0.patch(var_11, **var_12)
    var_14 = 'repo/'
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = True
    var_18 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 11/33 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 41 evaluates to True when chunk is not empty.'
    var_1 = 'clone'
    var_2 = 'http://example.com/repo.zip'
    var_3 = b'chunk1data'
    var_4 = b''
    var_5 = b'chunk3data'
    var_6 = 'test_dir/'
    var_7 = []
    var_8 = True
    var_9 = b'chunk1data'
    var_10 = bool(b'chunk1data' in var_7)
    assert var_10 is True
    var_11 = b'chunk3data'
    var_12 = bool(b'chunk3data' in var_7)
    assert var_12 is True
    var_13 = b''
    var_14 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_iter_content_chunk_predicate_evaluates_to_false. Retrieved 11/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks.'
    var_1 = b'data'
    var_2 = b''
    var_3 = b'more_data'
    var_4 = 'cookiecutter.zipfile.requests.get'
    var_5 = 'project/'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_8 = 'http://example.com/repo.zip'
    var_9 = True
    var_10 = 1024



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_valid_local_zipfile. Retrieved 10/26 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 5/15 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 7/17 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 6/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/25 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'Test unzip with a valid local zipfile.'
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'test content'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'clone'
    var_8 = True
    var_9 = False
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = True
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'empty'
    var_7 = bool('empty' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = True
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'top-level'
    var_9 = bool('top-level' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = True
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'not a valid zip archive'
    var_8 = bool('not a valid zip archive' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'test'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'nonexistent'
    var_8 = 'clone'
    var_9 = False

def test_case_0():
    var_0 = 'Test unzip expands user home directory.'
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'test'
    var_4 = 'test.zip'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'clone'
    var_8 = True
    var_9 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 11/23 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'test_dir/'
    var_4 = ''
    var_5 = 'test_dir/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.prompt_and_delete'
    var_8 = False
    var_9 = 'return_value'
    var_10 = {var_9: var_8}
    var_11 = module_0.patch(var_7, **var_10)
    var_12 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_download_predicate_false_when_reusing_existing. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when user chooses to reuse existing file.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = f'http://example.com/{var_1}'
    var_7 = True
    var_8 = False
    var_9 = 'test_project'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/21 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/21 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test unzipping a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project-name/'
    var_4 = ''
    var_5 = 'project-name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project-name'

def test_case_0():
    var_0 = 'Test unzipping an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'Test unzipping a zipfile without top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzipping an invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = 'This is not a valid zip file'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'new_clone_dir'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = bool(not var_5)
    assert var_7 is True
    var_8 = False

def test_case_0():
    var_0 = 'Test unzip expands user home directory.'
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = '.'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_download_predicate_false. Retrieved 17/33 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'clone'
    var_2 = 'http://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = False
    var_6 = 'return_value'
    var_7 = {var_6: var_5}
    var_8 = module_0.patch(var_4, **var_7)
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'project_dir/'
    var_13 = 'project_dir/file.txt'
    var_14 = 'cookiecutter.zipfile.requests.get'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = {}
    var_19 = module_0.patch(var_17, **var_18)
    var_20 = 'temp'
    var_21 = True



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 18/41 statements.
# Partially parsed test_unzip_with_local_file_extracts_zipfile. Retrieved 12/29 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_without_top_level_directory_raises_error. Retrieved 7/18 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_with_password_protected_zipfile. Retrieved 14/32 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 6/16 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a zipfile from a URL.'
    var_1 = 'content'
    var_2 = 'test_project/'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'
    var_6 = 'test_project/'
    var_7 = 'file.txt'
    var_8 = 'test_project/file.txt'
    var_9 = 'clone'
    var_10 = 'requests.get'
    var_11 = 'cookiecutter.zipfile.prompt_and_delete'
    var_12 = True
    var_13 = 'return_value'
    var_14 = {var_13: var_12}
    var_15 = module_0.patch(var_11, **var_14)
    var_16 = 'os.path.exists'
    var_17 = False
    var_18 = 'return_value'
    var_19 = {var_18: var_17}
    var_20 = module_0.patch(var_16, **var_19)
    var_21 = 'https://example.com/test.zip'
    var_22 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip extracts a local zipfile.'
    var_1 = 'content'
    var_2 = 'local_project/'
    var_3 = 'file.txt'
    var_4 = 'local content'
    var_5 = 'local.zip'
    var_6 = 'local_project/'
    var_7 = 'file.txt'
    var_8 = 'local_project/file.txt'
    var_9 = 'clone'
    var_10 = False
    var_11 = True
    var_12 = 'local_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zipfile lacks top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip extracts password-protected zipfile when password is provided.'
    var_1 = 'content'
    var_2 = 'protected_project/'
    var_3 = 'file.txt'
    var_4 = 'protected content'
    var_5 = 'protected.zip'
    var_6 = 'protected_project/'
    var_7 = 'file.txt'
    var_8 = 'protected_project/file.txt'
    var_9 = b'test_password'
    var_10 = 'clone'
    var_11 = False
    var_12 = True
    var_13 = 'test_password'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'content'
    var_2 = 'test_project/'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 14/45 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = 'clone'
    var_2 = b'test_data_1'
    var_3 = b'test_data_2'
    var_4 = 'project_name/'
    var_5 = []
    var_6 = 'requests.get'
    var_7 = 'builtins.open'
    var_8 = 'zipfile.ZipFile'
    var_9 = 'tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = 'http://example.com/archive.zip'
    var_12 = True
    var_13 = len(var_5)
    assert var_13 == 2
    var_14 = var_5[0]
    var_15 = bool(var_5[0] == var_2)
    assert var_15 is True
    var_16 = var_5[1]
    var_17 = bool(var_5[1] == var_3)
    assert var_17 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 10/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 7/16 statements.
# Partially parsed test_unzip_url_with_no_input. Retrieved 19/33 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/21 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 11/24 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 9/19 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = False
    var_11 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notopdir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = False
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'cookiecutter.zipfile.requests.get'
    var_16 = {}
    var_17 = module_0.patch(var_15, **var_16)
    var_18 = b'fake content'
    var_19 = [var_18]
    var_20 = 'builtins.open'
    var_21 = 'http://example.com/test.zip'
    var_22 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = 'cookiecutter.zipfile.ZipFile.extractall'
    var_9 = 'Bad password'
    var_10 = [var_9]
    var_11 = False
    var_12 = 'wrongpass'
    var_13 = bool(False)
    assert var_13 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = 'cookiecutter.zipfile.ZipFile.extractall'
    var_9 = 'Bad password'
    var_10 = [var_9]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unzip_predicate_line_54_evaluates_to_false. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This test ensures that when a zipfile contains at least one entry,\n    the condition evaluates to False and no InvalidZipRepository exception is raised.\n    '
    var_1 = 'test.zip'
    var_2 = 'project_root/'
    var_3 = ''
    var_4 = 'project_root/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = bool(var_5)
    assert var_9 is True
    var_10 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unzip_raises_error_when_zip_file_is_empty. Retrieved 7/18 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5, var_2)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'empty'
    var_9 = bool('empty' in str(e).lower())
    assert var_9 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unzip_with_url_no_existing_file. Retrieved 17/31 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 13/28 statements.
# Partially parsed test_unzip_empty_repository. Retrieved 7/14 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 8/15 statements.
# Partially parsed test_unzip_bad_zip_file. Retrieved 7/12 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 13/25 statements.
# Partially parsed test_unzip_password_protected_no_input. Retrieved 11/19 statements.
# Partially parsed test_unzip_password_protected_user_input. Retrieved 10/16 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip downloads and extracts a URL when file doesn't exist."
    var_1 = b'PK\x03\x04'
    var_2 = b'\x00'
    var_3 = 100
    var_4 = var_2 * var_3
    var_5 = var_1 + var_4
    var_6 = 'requests.get'
    var_7 = 'project/'
    var_8 = 'project/file.txt'
    var_9 = None
    var_10 = 'zipfile.ZipFile'
    var_11 = 'tempfile.mkdtemp'
    var_12 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_13 = {}
    var_14 = module_0.patch(var_12, **var_13)
    var_15 = 'http://example.com/project.zip'
    var_16 = True
    var_17 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'local.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'myproject/'
    var_4 = 'myproject/file.txt'
    var_5 = None
    var_6 = 'zipfile.ZipFile'
    var_7 = 'tempfile.mkdtemp'
    var_8 = 'temp'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = False
    var_13 = 'myproject'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = None
    var_5 = 'zipfile.ZipFile'
    var_6 = 'http://example.com/empty.zip'
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'empty'
    var_10 = bool('empty' in str(e).lower())
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = None
    var_6 = 'zipfile.ZipFile'
    var_7 = 'http://example.com/bad.zip'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'top-level'
    var_11 = bool('top-level' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for bad zip file.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = 'zipfile.ZipFile'
    var_5 = 'Bad zip'
    var_6 = [var_5]
    var_7 = 'http://example.com/bad.zip'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'not a valid zip archive'
    var_11 = bool('not a valid zip archive' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip extracts password-protected zip with provided password.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = 'encrypted'
    var_7 = [var_6]
    var_8 = None
    var_9 = 'zipfile.ZipFile'
    var_10 = 'tempfile.mkdtemp'
    var_11 = 'http://example.com/protected.zip'
    var_12 = False
    var_13 = 'mypassword'
    var_14 = 'project'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zip with no_input=True.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = 'encrypted'
    var_7 = [var_6]
    var_8 = None
    var_9 = 'zipfile.ZipFile'
    var_10 = 'http://example.com/protected.zip'
    var_11 = False
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'password'
    var_15 = bool('password' in str(e).lower())
    assert var_15 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip prompts user for password when needed.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = {}
    var_3 = module_0.patch(var_1, **var_2)
    var_4 = 'cookiecutter.zipfile.read_repo_password'
    var_5 = 'correct'
    var_6 = 'return_value'
    var_7 = {var_6: var_5}
    var_8 = module_0.patch(var_4, **var_7)
    var_9 = 'project/'
    var_10 = 'project/file.txt'
    var_11 = 'encrypted'
    var_12 = [var_11]
    var_13 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_with_url_no_input_no_existing_file. Retrieved 10/26 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 12/31 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 10/21 statements.
# Partially parsed test_unzip_password_protected_no_password_no_input_raises_error. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_top_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'
    var_8 = bool('top-level directory' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'
    var_7 = bool('not a valid zip archive' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip with URL when no existing file and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'rb'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'old_project/'
    var_8 = ''
    var_9 = 'rb'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password provided.'
    var_1 = 'protected.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = b'mypassword'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'mypassword'
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip, no password, and no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = b'mypassword'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'password protected'
    var_12 = bool('password protected' in str(e).lower())
    assert var_12 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 10/28 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/29 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'zipdir'
    var_2 = 'test_project'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'
    var_6 = 'file.txt'
    var_7 = 'test_project/file.txt'
    var_8 = 'clone'
    var_9 = False
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'
    var_8 = bool('top-level directory' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'
    var_7 = bool('not a valid zip archive' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'zipdir'
    var_2 = 'test_project'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'test.zip'
    var_6 = 'file.txt'
    var_7 = 'test_project/file.txt'
    var_8 = 'nonexistent'
    var_9 = 'clone'
    var_10 = False
    var_11 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises error for password protected zip with no_input=True.'
    var_1 = 'protected.zip'
    var_2 = 'test_project/file.txt'
    var_3 = 'content'
    var_4 = b'password'
    var_5 = 'clone'
    var_6 = False
    var_7 = True
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool('password' in str(e).lower() or 'protected' in str(e).lower())
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 13/29 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = True
    var_12 = False
    var_13 = None
    var_14 = 'project_dir'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_empty_zip_raises_invalid_zip_repository. Retrieved 8/16 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = 'cookiecutter.zipfile.os.path.exists'
    var_6 = False
    var_7 = 'return_value'
    var_8 = {var_7: var_6}
    var_9 = module_0.patch(var_5, **var_8)
    var_10 = False
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Zip repository'
    var_13 = 'is empty'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing version rather than re-download.\n    '
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'valid.zip'
    var_4 = 'test_project/'
    var_5 = ''
    var_6 = 'test_project/file.txt'
    var_7 = 'content'
    var_8 = 'test_project/'
    var_9 = 'test_project/file.txt'
    var_10 = True
    var_11 = False
    var_12 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 10/29 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This means the zip file contains at least one entry.\n    '
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = []
    var_7 = False
    var_8 = True
    var_9 = None
    var_10 = bool(var_5)
    assert var_10 is True
    var_11 = 'project_dir'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 11/27 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 6/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 12/24 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_top_level.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'bad.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = True
    var_11 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password protected zip and no_input raises error.'
    var_1 = 'protected.zip'
    var_2 = b'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test unzip with correct password extracts successfully.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'test_password'
    var_10 = 'project_name'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_predicate_line_54_evaluates_to_false. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = bool(var_4)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = len(var_2)
    var_7 = 0
    var_8 = var_6 == var_7
    assert var_8 is False
    var_9 = len(var_7)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 7/17 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zipfile is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5, var_2)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'empty'
    var_9 = bool('empty' in str(e).lower())
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unzip_predicate_line_39_evaluates_to_false. Retrieved 12/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = b''
    var_7 = None
    var_8 = b'some_data'
    var_9 = 'test_dir/'
    var_10 = 'test_dir/file.txt'
    var_11 = False
    var_12 = bool(var_8)
    assert var_12 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'empty'
    var_9 = bool('empty' in str(e).lower())
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 15/30 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/23 statements.
# Partially parsed test_unzip_empty_zip. Retrieved 11/23 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 12/24 statements.
# Partially parsed test_unzip_bad_zip_file. Retrieved 12/21 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 18/35 statements.
# Partially parsed test_unzip_password_protected_invalid_password. Retrieved 2/3 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip with URL when zip file doesn't exist yet."
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = b'test_chunk'
    var_4 = [var_3]
    var_5 = 'requests.get'
    var_6 = 'project_name/'
    var_7 = 'project_name/file.txt'
    var_8 = False
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = 'tempfile.mkdtemp'
    var_11 = 'temp'
    var_12 = 'os.path.exists'
    var_13 = 'return_value'
    var_14 = {var_13: var_8}
    var_15 = module_0.patch(var_12, **var_14)
    var_16 = True

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = 'repo.zip'
    var_2 = 'clone'
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = False
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = 'tempfile.mkdtemp'
    var_8 = 'temp'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with empty zip file raises InvalidZipRepository.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = b'test_chunk'
    var_4 = [var_3]
    var_5 = 'requests.get'
    var_6 = False
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = 'os.path.exists'
    var_9 = 'return_value'
    var_10 = {var_9: var_6}
    var_11 = module_0.patch(var_8, **var_10)
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip when zip doesn't have top-level directory raises InvalidZipRepository."
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = b'test_chunk'
    var_4 = [var_3]
    var_5 = 'requests.get'
    var_6 = 'file.txt'
    var_7 = False
    var_8 = 'cookiecutter.zipfile.ZipFile'
    var_9 = 'os.path.exists'
    var_10 = 'return_value'
    var_11 = {var_10: var_7}
    var_12 = module_0.patch(var_9, **var_11)
    var_13 = True
    var_14 = bool(False)
    assert var_14 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = b'test_chunk'
    var_4 = [var_3]
    var_5 = 'requests.get'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = 'Invalid zip'
    var_8 = [var_7]
    var_9 = 'os.path.exists'
    var_10 = False
    var_11 = 'return_value'
    var_12 = {var_11: var_10}
    var_13 = module_0.patch(var_9, **var_12)
    var_14 = True
    var_15 = bool(False)
    assert var_15 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and provided password.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = b'test_chunk'
    var_4 = [var_3]
    var_5 = 'requests.get'
    var_6 = 'project_name/'
    var_7 = 'project_name/file.txt'
    var_8 = 'Bad password'
    var_9 = [var_8]
    var_10 = None
    var_11 = False
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = 'tempfile.mkdtemp'
    var_14 = 'temp'
    var_15 = 'os.path.exists'
    var_16 = 'return_value'
    var_17 = {var_16: var_11}
    var_18 = module_0.patch(var_15, **var_17)
    var_19 = True
    var_20 = 'secret'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and invalid password raises error.'
    var_1 = 'https://example.com/repo.zip'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unzip_predicate_at_line_40_evaluates_to_false. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks.'
    var_1 = b''
    var_2 = None
    var_3 = [var_1, var_1, var_2]
    var_4 = 'test.zip'
    var_5 = 'project_name/'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = 1024



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_with_url_downloads_file. Retrieved 10/27 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 10/22 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 9/25 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 9/21 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level'
    var_8 = bool('top-level' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = 'not a zip file'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'
    var_7 = bool('not a valid zip archive' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip downloads file when is_url is True.'
    var_1 = 'clone'
    var_2 = 'temp.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'http://example.com/project.zip'
    var_8 = True
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected file with no_input.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/'
    var_6 = 'Bad password'
    var_7 = [var_6]
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'password'
    var_12 = bool('password' in str(e).lower())
    assert var_12 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'new_clone_dir'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = bool(not var_5)
    assert var_7 is True
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip expands user home directory in clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unzip_iter_content_filters_empty_chunks. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for empty chunks.'
    var_1 = 'http://example.com/test.zip'
    var_2 = b'chunk1'
    var_3 = b''
    var_4 = b'chunk2'
    var_5 = 'requests.get'
    var_6 = 'builtins.open'
    var_7 = 'project/'
    var_8 = 'zipfile.ZipFile'
    var_9 = 'tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 15/36 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 15/30 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 9/19 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 11/20 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 10/17 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 16/30 statements.
# Partially parsed test_unzip_url_existing_file_no_input. Retrieved 17/39 statements.


def test_case_0():
    var_0 = 'Test unzip with a URL to a new zipfile.'
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = 'temp'
    var_9 = 'cookiecutter.zipfile'
    var_10 = 'unzip'
    var_11 = [var_10]
    var_12 = __import__(var_9, fromlist=var_11)
    var_13 = 'https://example.com/test.zip'
    var_14 = True
    var_15 = 'test_project'
    var_16 = bool(var_14)
    assert var_16 is True

def test_case_0():
    var_0 = 'Test unzip with a local zipfile path.'
    var_1 = 'test.zip'
    var_2 = 'local_project/'
    var_3 = ''
    var_4 = 'local_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'temp'
    var_8 = 'cookiecutter.zipfile'
    var_9 = 'unzip'
    var_10 = [var_9]
    var_11 = __import__(var_8, fromlist=var_10)
    var_12 = False
    var_13 = True
    var_14 = 'local_project'
    var_15 = bool(var_10)
    assert var_15 is True

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile'
    var_4 = 'unzip'
    var_5 = [var_4]
    var_6 = __import__(var_3, fromlist=var_5)
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'empty'
    var_11 = bool('empty' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 'Test unzip with zipfile missing top-level directory raises InvalidZipRepository.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile'
    var_6 = 'unzip'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = False
    var_10 = True
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzip with an invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile'
    var_5 = 'unzip'
    var_6 = [var_5]
    var_7 = __import__(var_4, fromlist=var_6)
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'not a valid zip archive'

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'protected_project/'
    var_3 = ''
    var_4 = 'protected_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'temp'
    var_8 = 'cookiecutter.zipfile'
    var_9 = 'unzip'
    var_10 = [var_9]
    var_11 = __import__(var_8, fromlist=var_10)
    var_12 = False
    var_13 = True
    var_14 = 'test123'
    var_15 = 'protected_project'
    var_16 = bool(var_10)
    assert var_16 is True

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True (should delete and redownload).'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'old content'
    var_4 = []
    var_5 = 'new_project/'
    var_6 = ''
    var_7 = 'new_project/file.txt'
    var_8 = 'new content'
    var_9 = 0
    var_10 = 'temp'
    var_11 = 'cookiecutter.zipfile'
    var_12 = 'unzip'
    var_13 = [var_12]
    var_14 = __import__(var_11, fromlist=var_13)
    var_15 = 'https://example.com/test.zip'
    var_16 = True
    var_17 = 'new_project'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_unzip_zipfile_context_manager_line_54. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = 'namelist'
    var_7 = bool(var_3)
    assert var_7 is True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = bool(var_5 > 0)
    assert var_9 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_zipfile_predicate_line_54_evaluates_to_false. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This test ensures that when a zipfile contains at least one entry,\n    the condition len(zip_file.namelist()) == 0 is False.\n    '
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = var_2 == var_6
    assert var_7 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_unzip_context_manager_with_zipfile. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = bool(var_5)
    assert var_9 is True
    var_10 = 'test_project'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for keep-alive chunks.'
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'some data'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unzip_context_manager_with_zipfile. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (with ZipFile(zip_path) as zip_file:) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = 'test_project'
    var_10 = True



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 7/24 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/19 statements.
# Partially parsed test_unzip_empty_repository_raises_error. Retrieved 3/18 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/20 statements.
# Partially parsed test_unzip_with_password. Retrieved 10/27 statements.
# Partially parsed test_unzip_invalid_password_raises_error. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = b'test_data'
    var_2 = [var_1]
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = True
    var_6 = 'project_name'
    var_7 = bool(var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'local.zip'
    var_1 = 'project/'
    var_2 = 'project/file.txt'
    var_3 = False
    var_4 = 'project'
    var_5 = bool(var_2)
    assert var_5 is True

def test_case_0():
    var_0 = 'https://example.com/empty.zip'
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'https://example.com/notoplevel.zip'
    var_1 = b'data'
    var_2 = [var_1]
    var_3 = 'file.txt'
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = 'secret'
    var_2 = b'data'
    var_3 = [var_2]
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = 'Bad password'
    var_7 = [var_6]
    var_8 = None
    var_9 = True
    var_10 = 'project'
    var_11 = bool(var_4)
    assert var_11 is True

def test_case_0():
    var_0 = 'https://example.com/protected.zip'
    var_1 = 'wrongpassword'
    var_2 = b'data'
    var_3 = [var_2]
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = 'Bad password'
    var_7 = [var_6]
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    pass



