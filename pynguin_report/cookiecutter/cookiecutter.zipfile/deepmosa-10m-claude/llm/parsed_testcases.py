####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file_success. Retrieved 7/19 statements.
# Partially parsed test_unzip_url_new_file. Retrieved 10/27 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 8/20 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 9/23 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'extract'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'project_name'

def test_case_0():
    var_0 = 'clone'
    var_1 = b'test'
    var_2 = [var_1]
    var_3 = 'temp.zip'
    var_4 = 'test_project/'
    var_5 = ''
    var_6 = 'test_project/file.txt'
    var_7 = 'content'
    var_8 = 'http://example.com/test_project.zip'
    var_9 = True
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'top-level'
    var_7 = bool('top-level' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'valid zip archive'
    var_6 = bool('valid zip archive' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'test'
    var_8 = 'project'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'protected.zip'
    var_2 = b'wrongpass'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'password'
    var_11 = bool('password' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/18 statements.


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
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'InvalidZipRepository'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'new_clone'
    var_7 = False
    var_8 = 'project_name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when is_url is True, the zip file exists, and prompt_and_delete\n    returns False (user chooses to reuse existing version).\n    '
    var_1 = 'http://example.com/test.zip'
    var_2 = 'test.zip'
    var_3 = b'dummy'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = 'get'
    var_6 = 0
    var_7 = {var_5: var_6}
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'project_name/'
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = True
    var_12 = False
    var_13 = var_7['get']
    assert var_13 == 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/23 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/23 statements.


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
    var_0 = 'Test unzip with empty zipfile raises InvalidZipRepository.'
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
    var_0 = 'Test unzip with password-protected zip and no_input raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = b'password'
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
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test'
    var_9 = 'project_name'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_empty_zipfile_predicate_false. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 evaluates to False when zipfile is empty.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'with_content.zip'
    var_4 = 'project/'
    var_5 = ''
    var_6 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 10/20 statements.
# Partially parsed test_unzip_with_password_protected_file_no_input. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 9/19 statements.
# Partially parsed test_unzip_with_invalid_password_provided. Retrieved 9/21 statements.


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
    var_8 = bool('top-level directory' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
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
    var_0 = 'Test unzip with password protected file and no_input raises error.'
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
    var_11 = 'password protected'
    var_12 = bool('password protected' in str(e).lower())
    assert var_12 is True

def test_case_0():
    var_0 = 'Test unzip with password protected file and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test_password'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password protected file and invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'wrong_password'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid password'
    var_11 = bool('invalid password' in str(e).lower())
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 22/35 statements.
# Partially parsed test_unzip_with_local_file_extracts_zipfile. Retrieved 13/22 statements.
# Partially parsed test_unzip_existing_file_prompts_for_deletion. Retrieved 22/33 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 8/17 statements.
# Partially parsed test_unzip_missing_top_level_directory_raises_error. Retrieved 9/18 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/15 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a URL-based zipfile.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.prompt_and_delete'
    var_10 = True
    var_11 = 'return_value'
    var_12 = {var_11: var_10}
    var_13 = module_0.patch(var_9, **var_12)
    var_14 = 'cookiecutter.zipfile.ZipFile'
    var_15 = {}
    var_16 = module_0.patch(var_14, **var_15)
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = '/tmp/test'
    var_19 = 'return_value'
    var_20 = {var_19: var_18}
    var_21 = module_0.patch(var_17, **var_20)
    var_22 = 'project_name/'
    var_23 = 'project_name/file.txt'
    var_24 = b'chunk1'
    var_25 = b'chunk2'
    var_26 = 'builtins.open'
    var_27 = False
    var_28 = 100

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip extracts a local zipfile without downloading.'
    var_1 = 'local.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = '/tmp/test'
    var_11 = 'return_value'
    var_12 = {var_11: var_10}
    var_13 = module_0.patch(var_9, **var_12)
    var_14 = 'my_project/'
    var_15 = 'my_project/setup.py'
    var_16 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip prompts to delete existing cached zipfile.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
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
    var_21 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_22 = '/tmp/test'
    var_23 = 'return_value'
    var_24 = {var_23: var_22}
    var_25 = module_0.patch(var_21, **var_24)
    var_26 = 'project/'
    var_27 = 'project/file.txt'
    var_28 = b'chunk'
    var_29 = 'builtins.open'
    var_30 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'empty'
    var_12 = bool('empty' in str(e).lower())
    assert var_12 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when top-level is not a directory.'
    var_1 = 'bad.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'file.txt'
    var_10 = False
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'top-level directory'
    var_13 = bool('top-level directory' in str(e).lower())
    assert var_13 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip extracts password-protected zipfile with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'clone'
    var_3 = 'mypassword'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_11 = '/tmp/test'
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception_handling. Retrieved 7/19 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = 'this is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = module_0.unzip(var_0, var_4, var_2, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'is not a valid zip archive'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/19 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 10/20 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 11/23 statements.
# Partially parsed test_unzip_password_protected_with_wrong_password_raises_error. Retrieved 12/25 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 12/25 statements.


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
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory exists.'
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
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a valid zip'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'not a valid zip archive'
    var_7 = bool('not a valid zip archive' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = 'test_project'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False
    var_11 = 'test_project'
    var_12 = bool(var_7)
    assert var_12 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False
    var_11 = 'wrong_password'
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'invalid password'
    var_14 = bool('invalid password' in str(e).lower())
    assert var_14 is True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False
    var_11 = True
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'unable to unlock'
    var_14 = bool('unable to unlock' in str(e).lower())
    assert var_14 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 15/37 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when is_url is True, zip_path exists, and prompt_and_delete returns False.\n    '
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = b'dummy content'
    var_5 = 'cookiecutter.zipfile.prompt_and_delete'
    var_6 = 'get'
    var_7 = 0
    var_8 = {var_6: var_7}
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = 'project/'
    var_11 = None
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = True
    var_14 = False
    var_15 = var_8['get']
    assert var_15 == 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 7/18 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_bad_zip_file_raises_invalid_zip_repository. Retrieved 8/17 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = module_0.unzip(var_0, var_6, var_2)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'is not a valid zip archive'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_opens_zipfile_with_context_manager. Retrieved 9/26 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_download_predicate_false. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 11/32 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/27 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/18 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/19 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 12/28 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 9/22 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = "Test unzip with a URL when the zip file doesn't exist locally."
    var_1 = 'clone'
    var_2 = b'PK\x03\x04'
    var_3 = [var_2]
    var_4 = 'project_name/'
    var_5 = 'project_name/file.txt'
    var_6 = None
    var_7 = 'http://example.com/project.zip'
    var_8 = True
    var_9 = 'temp'
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'clone'
    var_2 = 'local.zip'
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = None
    var_6 = False
    var_7 = True
    var_8 = 'temp'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zip is empty.'
    var_1 = 'clone'
    var_2 = None
    var_3 = 'local.zip'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zip has no top-level directory.'
    var_1 = 'clone'
    var_2 = 'file.txt'
    var_3 = None
    var_4 = 'local.zip'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password provided.'
    var_1 = 'clone'
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = 'Bad password'
    var_5 = [var_4]
    var_6 = None
    var_7 = 'local.zip'
    var_8 = False
    var_9 = True
    var_10 = 'mypassword'
    var_11 = 'temp'
    var_12 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'clone'
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = 'Bad password'
    var_5 = [var_4]
    var_6 = None
    var_7 = 'local.zip'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'clone'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 6/15 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.requests.get'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = False
    var_6 = module_1.unzip(var_0, var_5, var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 14/38 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 12/31 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 7/25 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 9/27 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 9/22 statements.


def test_case_0():
    var_0 = "Test unzip with URL when zip file doesn't exist yet."
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = 'temp'
    var_9 = True
    var_10 = 'clone'
    var_11 = 'test.zip'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = None

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'test.zip'
    var_8 = 'clone'
    var_9 = 'temp'
    var_10 = True
    var_11 = False
    var_12 = None
    var_13 = 'test_project'

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = []
    var_2 = 0
    var_3 = 'empty.zip'
    var_4 = 'clone'
    var_5 = False
    var_6 = True
    var_7 = None
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip has no top-level directory.'
    var_1 = []
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 0
    var_5 = 'notoplevel.zip'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = None
    var_10 = bool(False)
    assert var_10 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = b'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = module_0.unzip(var_0, var_4, var_2, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = []
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 8
    var_7 = 0
    var_8 = 'protected.zip'
    var_9 = 'clone'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_local_file_valid_zip. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/21 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 9/21 statements.
# Partially parsed test_unzip_password_protected_no_input_raises. Retrieved 9/20 statements.
# Partially parsed test_unzip_password_protected_with_wrong_password. Retrieved 9/20 statements.
# Partially parsed test_unzip_returns_unzip_path. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'Test unzipping a local valid zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzipping an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'empty'
    var_6 = bool('empty' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzipping a zipfile without top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'Test unzipping an invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'valid zip archive'

import genericpath as module_0

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
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
    var_0 = 'Test unzipping a password-protected zipfile with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzipping password-protected zipfile with no_input=True raises.'
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

def test_case_0():
    var_0 = 'Test unzipping password-protected zipfile with wrong password raises.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'wrongpass'
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'Test that unzip returns the correct unzip_path.'
    var_1 = 'test.zip'
    var_2 = 'my_project/'
    var_3 = ''
    var_4 = 'my_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'my_project'



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to False for empty chunks.'
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_download_predicate_false_when_reusing_existing. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when user chooses to reuse existing file.'
    var_1 = 'existing_archive.zip'
    var_2 = f'https://example.com/{var_1}'
    var_3 = 'project_dir/'
    var_4 = 'project_dir/file.txt'
    var_5 = True
    var_6 = False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 5/15 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that predicate at line 55 evaluates to True when zipfile is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'is empty'



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 14/46 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'chunk1data'
    var_2 = b''
    var_3 = b'chunk3data'
    var_4 = 'test.zip'
    var_5 = 'project_dir/'
    var_6 = ''
    var_7 = 'project_dir/file.txt'
    var_8 = 'content'
    var_9 = []
    var_10 = 'project_dir/'
    var_11 = 'project_dir/file.txt'
    var_12 = 'http://example.com/test.zip'
    var_13 = True
    var_14 = bool(var_1 in var_9)
    assert var_14 is True
    var_15 = bool(var_3 in var_9)
    assert var_15 is True
    var_16 = bool(var_2 not in var_9)
    assert var_16 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 6/24 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 5/18 statements.
# Partially parsed test_unzip_empty_repository_raises_error. Retrieved 3/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/17 statements.
# Partially parsed test_unzip_with_password. Retrieved 8/23 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 3/14 statements.
# Partially parsed test_unzip_prompt_and_delete_existing_file. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test unzip downloads a URL and extracts it successfully.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'test_project/'
    var_3 = 'test_project/file.txt'
    var_4 = b'test'
    var_5 = True

def test_case_0():
    var_0 = 'Test unzip uses a local file without downloading.'
    var_1 = '/local/path/repo.zip'
    var_2 = 'test_project/'
    var_3 = 'test_project/file.txt'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'https://example.com/empty.zip'
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'file.txt'
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'Test unzip extracts password-protected zip with provided password.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'test_password'
    var_3 = 'test_project/'
    var_4 = 'test_project/file.txt'
    var_5 = 'Bad password'
    var_6 = [var_5]
    var_7 = None
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'https://example.com/invalid.zip'
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test unzip prompts to delete existing zip file.'
    var_1 = 'https://example.com/repo.zip'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'chunk1'
    var_2 = b'chunk2'
    var_3 = b''
    var_4 = [var_1, var_2, var_3]
    var_5 = 'http://example.com/test.zip'
    var_6 = 'test.zip'
    var_7 = True
    var_8 = 1024
    var_9 = var_2.write.call_count
    assert var_9 == 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_with_valid_local_zipfile. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_url_new_file. Retrieved 17/34 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 7/15 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 12/23 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 12/24 statements.
# Partially parsed test_unzip_with_url_existing_file_prompts_delete. Retrieved 9/16 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a valid local zipfile.'
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
    var_0 = "Test unzip with URL when zipfile doesn't exist yet."
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
    var_11 = False
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'rb'
    var_16 = 'requests.get'
    var_17 = 'builtins.open'
    var_18 = 'http://example.com/test.zip'
    var_19 = True
    var_20 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
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
    var_1 = 'no_top_level.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = {}
    var_7 = module_0.patch(var_5, **var_6)
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'top-level'
    var_11 = bool('top-level' in str(e).lower())
    assert var_11 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'clone'
    var_2 = 'fake.zip'
    var_3 = 'not a zip file'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'valid zip'
    var_10 = bool('valid zip' in str(e).lower())
    assert var_10 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'password'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = 'password'
    var_13 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'password'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = False
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'password'
    var_15 = bool('password' in str(e).lower())
    assert var_15 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when zipfile exists prompts to delete.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/21 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 9/21 statements.
# Partially parsed test_unzip_expanduser_in_clone_to_dir. Retrieved 8/21 statements.


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
    var_0 = 'Test unzip with empty zipfile raises InvalidZipRepository.'
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
    var_0 = 'Test unzip with invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'valid zip'
    var_7 = bool('valid zip' in str(e).lower())
    assert var_7 is True

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
    var_0 = 'Test unzip with password protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = b'password'
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
    var_0 = 'Test unzip with password protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test'

def test_case_0():
    var_0 = 'Test unzip expands user home directory in clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'project_name'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_zipfile_predicate_line_54_evaluates_to_false. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = len(var_2)
    var_7 = 0
    var_8 = var_6 == var_7
    assert var_8 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_local_file_valid_zip. Retrieved 13/29 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 8/16 statements.
# Partially parsed test_unzip_url_downloads_and_extracts. Retrieved 18/36 statements.
# Partially parsed test_unzip_url_prompts_to_delete_existing. Retrieved 18/39 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 14/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a local zipfile with valid structure.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_11 = 'temp'
    var_12 = False
    var_13 = True
    var_14 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that empty zipfile raises InvalidZipRepository.'
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
    var_0 = 'Test that zip without top-level directory raises InvalidZipRepository.'
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
    var_11 = 'top-level'
    var_12 = bool('top-level' in str(e).lower())
    assert var_12 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that invalid zip file raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a valid zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = False
    var_8 = True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'not a valid zip archive'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping from URL downloads and extracts properly.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = [var_5]
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.os.path.exists'
    var_12 = False
    var_13 = 'return_value'
    var_14 = {var_13: var_12}
    var_15 = module_0.patch(var_11, **var_14)
    var_16 = 'cookiecutter.zipfile.requests.get'
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = 'temp'
    var_19 = 'http://example.com/test.zip'
    var_20 = True
    var_21 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that existing zip file prompts for deletion.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'old content'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = {}
    var_10 = module_0.patch(var_8, **var_9)
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = True
    var_14 = 'return_value'
    var_15 = {var_14: var_13}
    var_16 = module_0.patch(var_12, **var_15)
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = 'temp'
    var_19 = 'http://example.com/test.zip'
    var_20 = False
    var_21 = 'project_name'

import email._encoded_words as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test extracting password-protected zip with valid password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'utf-8'
    var_4 = module_0.encode(var_3)
    var_5 = 'project_name/'
    var_6 = ''
    var_7 = 'project_name/file.txt'
    var_8 = 'content'
    var_9 = 'clone'
    var_10 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_11 = {}
    var_12 = module_1.patch(var_10, **var_11)
    var_13 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_14 = 'temp'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filtering. Retrieved 11/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = 'http://example.com/test.zip'
    var_2 = b'chunk1'
    var_3 = b''
    var_4 = b'chunk2'
    var_5 = b'PK\x03\x04'
    var_6 = 'test_dir/'
    var_7 = True
    var_8 = None
    var_9 = b'chunk1'
    var_10 = b'chunk2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_downloads_zipfile_with_chunks. Retrieved 12/34 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = 'repo.zip'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = b'chunk3'
    var_5 = b''
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = 'project_dir/'
    var_8 = None
    var_9 = True
    var_10 = None
    var_11 = 'wb'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_context_manager_predicate. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (ZipFile context manager) evaluates to True.'
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_bad_zip_file_raises_invalid_zip_repository. Retrieved 10/22 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'clone'
    var_2 = 'bad.zip'
    var_3 = b'This is not a valid zip file'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = 'Bad zip file'
    var_9 = [var_8]
    var_10 = False
    var_11 = True
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'not a valid zip archive'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 12/25 statements.
# Partially parsed test_unzip_url_no_input_first_download. Retrieved 19/42 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 10/21 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 9/18 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 13/26 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 17/29 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = False
    var_12 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True for first download.'
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
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = 'cookiecutter.zipfile.requests.get'
    var_18 = 'builtins.open'
    var_19 = 'http://example.com/project.zip'
    var_20 = True
    var_21 = 'project_name'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = {}
    var_4 = module_0.patch(var_2, **var_3)
    var_5 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_6 = 'temp'
    var_7 = False
    var_8 = module_1.unzip(var_0, var_7, var_2)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'empty'
    var_11 = bool('empty' in str(e).lower())
    assert var_11 is True

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = {}
    var_6 = module_0.patch(var_4, **var_5)
    var_7 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_8 = 'temp'
    var_9 = False
    var_10 = module_1.unzip(var_2, var_9, var_4)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'top-level'
    var_13 = bool('top-level' in str(e).lower())
    assert var_13 is True

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = {}
    var_5 = module_0.patch(var_3, **var_4)
    var_6 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_7 = 'temp'
    var_8 = False
    var_9 = module_1.unzip(var_0, var_8, var_2)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'valid zip'
    var_12 = bool('valid zip' in str(e).lower())
    assert var_12 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password protected zipfile and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = False
    var_12 = 'testpass'
    var_13 = 'project_name'

import requests.api as module_0
import locale as module_1

def test_case_0():
    var_0 = 'Test unzip with password protected zipfile and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = b'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_11 = 'temp'
    var_12 = module_1.str(var_6)
    var_13 = 'return_value'
    var_14 = {var_13: var_12}
    var_15 = module_0.patch(var_10, **var_14)
    var_16 = False
    var_17 = 'project_name/'
    var_18 = 'project_name/file.txt'
    var_19 = [var_17, var_18]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_invalid_zip_file_raises_invalid_zip_repository. Retrieved 9/21 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'cookiecutter.zipfile.ZipFile'
    var_4 = 'Bad zip file'
    var_5 = [var_4]
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = False
    var_10 = module_1.unzip(var_0, var_9, var_2)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'is not a valid zip archive'



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False."
    var_1 = b''
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 16/34 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This happens when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing version.\n    '
    var_1 = 'clone'
    var_2 = 'http://example.com/archive.zip'
    var_3 = 'archive.zip'
    var_4 = b'dummy content'
    var_5 = 'cookiecutter.zipfile.prompt_and_delete'
    var_6 = False
    var_7 = 'return_value'
    var_8 = {var_7: var_6}
    var_9 = module_0.patch(var_5, **var_8)
    var_10 = 'project/'
    var_11 = None
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_14 = 'temp'
    var_15 = 'cookiecutter.zipfile.requests.get'
    var_16 = {}
    var_17 = module_0.patch(var_15, **var_16)
    var_18 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_predicate_at_line_36_evaluates_to_false. Retrieved 15/35 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.'
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = False
    var_6 = 'return_value'
    var_7 = {var_6: var_5}
    var_8 = module_0.patch(var_4, **var_7)
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = {}
    var_11 = module_0.patch(var_9, **var_10)
    var_12 = 'project_name/'
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = 'temp'
    var_16 = True
    var_17 = 'project_name'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 evaluates to False when zipfile is not empty.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = len(var_2)
    var_7 = 0
    var_8 = var_6 == var_7
    assert var_8 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_with_url_no_existing_file. Retrieved 9/28 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 7/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/22 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/23 statements.
# Partially parsed test_unzip_with_password. Retrieved 8/27 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 4/14 statements.


def test_case_0():
    var_0 = "Test unzip with a URL when file doesn't exist locally."
    var_1 = 'clone'
    var_2 = b'test_chunk'
    var_3 = [var_2]
    var_4 = 'temp'
    var_5 = True
    var_6 = 'project_name/'
    var_7 = 'project_name/file.txt'
    var_8 = 'http://example.com/test.zip'
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'temp'
    var_3 = True
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = False
    var_7 = 'project'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zip file.'
    var_1 = 'clone'
    var_2 = 'temp'
    var_3 = True
    var_4 = 'http://example.com/test.zip'
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'clone'
    var_2 = 'temp'
    var_3 = True
    var_4 = 'file.txt'
    var_5 = 'http://example.com/test.zip'
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected archive.'
    var_1 = 'test.zip'
    var_2 = 'temp'
    var_3 = True
    var_4 = 'project/'
    var_5 = 'project/file.txt'
    var_6 = False
    var_7 = 'testpass'
    var_8 = 'project'

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip file.'
    var_1 = 'clone'
    var_2 = 'temp'
    var_3 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/23 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/19 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 10/21 statements.
# Partially parsed test_unzip_with_expanduser_path. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'extract'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'test_project'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'empty'
    var_5 = bool('empty' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'clone'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'top-level directory'
    var_7 = bool('top-level directory' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = 'clone'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'not a valid zip archive'
    var_6 = bool('not a valid zip archive' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = 'nested'
    var_7 = False
    var_8 = 'test_project'

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
    var_0 = 'test.zip'
    var_1 = 'test_project/'
    var_2 = ''
    var_3 = 'test_project/file.txt'
    var_4 = 'content'
    var_5 = 'clone'
    var_6 = False
    var_7 = 'test_project'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 15/27 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'cookiecutter.zipfile.prompt_and_delete'
    var_3 = False
    var_4 = 'return_value'
    var_5 = {var_4: var_3}
    var_6 = module_0.patch(var_2, **var_5)
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = {}
    var_9 = module_0.patch(var_7, **var_8)
    var_10 = 'cookiecutter.zipfile.requests.get'
    var_11 = {}
    var_12 = module_0.patch(var_10, **var_11)
    var_13 = 'project_name/'
    var_14 = 'cookiecutter.zipfile.ZipFile'
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = 'http://example.com/repo.zip'
    var_18 = True



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for empty chunks.'
    var_1 = b''
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_when_zip_is_empty. Retrieved 7/18 statements.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 9/22 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 18/38 statements.


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
    var_8 = b'test content'
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = 'project/'
    var_11 = 'cookiecutter.zipfile.ZipFile'
    var_12 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_13 = 'temp'
    var_14 = 'http://example.com/test.zip'
    var_15 = False
    var_16 = 'cookiecutter.zipfile'
    var_17 = 'prompt_and_delete'
    var_18 = [var_17]
    var_19 = __import__(var_16, fromlist=var_18)
    var_20 = [var_19, var_17]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception_handling. Retrieved 8/23 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = 'this is not a valid zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = False
    var_6 = True
    var_7 = module_0.unzip(var_0, var_5, var_2, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'is not a valid zip archive'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_with_url_creates_directory_and_extracts. Retrieved 10/34 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 8/27 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/23 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/24 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'clone'
    var_1 = 'https://example.com/repo.zip'
    var_2 = b'PK\x03\x04'
    var_3 = [var_2]
    var_4 = 'project_name/'
    var_5 = 'project_name/file.txt'
    var_6 = 'extract'
    var_7 = True
    var_8 = 'project_name'
    var_9 = 100

def test_case_0():
    var_0 = 'clone'
    var_1 = 'local.zip'
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = 'extract'
    var_5 = True
    var_6 = False
    var_7 = 'project_name'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'https://example.com/empty.zip'
    var_2 = b'PK\x03\x04'
    var_3 = [var_2]
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'empty'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'https://example.com/bad.zip'
    var_2 = b'PK\x03\x04'
    var_3 = [var_2]
    var_4 = 'file.txt'
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'top-level directory'

def test_case_0():
    var_0 = 'clone'
    var_1 = 'https://example.com/protected.zip'
    var_2 = 'secret123'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 13/46 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 12/32 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 5/17 statements.
# Partially parsed test_unzip_missing_top_level_directory_raises_error. Retrieved 7/19 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 13/35 statements.


def test_case_0():
    var_0 = 'Test unzip downloads and extracts a zipfile from URL.'
    var_1 = 'zip_source'
    var_2 = 'test_project/'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'
    var_6 = 'test_project/'
    var_7 = 'file.txt'
    var_8 = 'test_project/file.txt'
    var_9 = 'clone'
    var_10 = []
    var_11 = 'get'
    var_12 = True
    var_13 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with a local zipfile path.'
    var_1 = 'zip_source'
    var_2 = 'test_project/'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'
    var_6 = 'test_project/'
    var_7 = 'file.txt'
    var_8 = 'test_project/file.txt'
    var_9 = 'clone'
    var_10 = False
    var_11 = True
    var_12 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'empty'
    var_7 = bool('empty' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zipfile lacks top-level directory.'
    var_1 = 'no_top_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'top-level directory'
    var_9 = bool('top-level directory' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'not a valid zip archive'
    var_8 = bool('not a valid zip archive' in str(e).lower())
    assert var_8 is True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'zip_source'
    var_2 = 'test_project/'
    var_3 = 'file.txt'
    var_4 = 'test content'
    var_5 = 'test.zip'
    var_6 = 'test_project/'
    var_7 = 'file.txt'
    var_8 = 'test_project/file.txt'
    var_9 = 'nonexistent'
    var_10 = 'clone'
    var_11 = False
    var_12 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 12/23 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 10/22 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 11/25 statements.


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
    var_1 = 'no_top_dir.zip'
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

import genericpath as module_0

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = module_0.exists(var_8)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = 'project'

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zip with no_input.'
    var_1 = 'protected.zip'
    var_2 = b'password'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool('password protected' in str(e).lower() or 'unable to unlock' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = b'mypassword'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'mypassword'
    var_10 = 'project'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_with_local_zipfile. Retrieved 12/24 statements.
# Partially parsed test_unzip_url_with_no_existing_file. Retrieved 17/32 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 8/16 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 14/27 statements.
# Partially parsed test_unzip_password_protected_with_invalid_password. Retrieved 13/25 statements.
# Partially parsed test_unzip_expanduser_in_clone_to_dir. Retrieved 10/19 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile that is not a URL.'
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
    var_0 = "Test unzip with URL when file doesn't exist yet."
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
    var_11 = False
    var_12 = 'return_value'
    var_13 = {var_12: var_11}
    var_14 = module_0.patch(var_10, **var_13)
    var_15 = 'rb'
    var_16 = 'requests.get'
    var_17 = 'http://example.com/test.zip'
    var_18 = True
    var_19 = 'project_name'

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
    var_0 = "Test unzip raises error when zip doesn't have top-level directory."
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
    var_0 = 'Test unzip raises error for invalid zip file.'
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
    var_0 = 'Test unzip with password-protected zipfile using correct password.'
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
    var_13 = 'test_password'
    var_14 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises error with invalid password for protected zipfile.'
    var_1 = 'protected.zip'
    var_2 = b'correct_password'
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
    var_13 = 'wrong_password'
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'invalid password'
    var_16 = bool('invalid password' in str(e).lower())
    assert var_16 is True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip expands ~ in clone_to_dir path.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = {}
    var_8 = module_0.patch(var_6, **var_7)
    var_9 = 'pathlib.Path.expanduser'
    var_10 = 'clone'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_context_manager_closes_zipfile. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (ZipFile context manager) evaluates to True.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = 'test_project'
    var_10 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 14/36 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = 'https://example.com/test.zip'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = b''
    var_5 = [var_2, var_3, var_4]
    var_6 = 'project_dir/'
    var_7 = 'project_dir/file.txt'
    var_8 = [var_6, var_7]
    var_9 = None
    var_10 = True
    var_11 = b'chunk1'
    var_12 = b'chunk2'
    var_13 = var_9.write.call_count
    assert var_13 == 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unzip_predicate_line_55_false. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 evaluates to False when zip file has content.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = bool(var_4)
    assert var_8 is True
    var_9 = bool(var_5 > 0)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unzip_context_manager_closes_zipfile. Retrieved 9/27 statements.


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
    var_10 = 'test_project'
    var_11 = True



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = b'x'
    var_2 = 1024
    var_3 = var_1 * var_2
    var_4 = b''
    var_5 = bool(var_3)
    assert var_5 is True
    var_6 = bool(not var_4)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 7/32 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/25 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 4/21 statements.
# Partially parsed test_unzip_missing_top_level_directory_raises_error. Retrieved 5/22 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 3/22 statements.


def test_case_0():
    var_0 = b'test_data'
    var_1 = [var_0]
    var_2 = 'test_project/'
    var_3 = 'test_project/file.txt'
    var_4 = 'http://example.com/test.zip'
    var_5 = True
    var_6 = 'test_project'

def test_case_0():
    var_0 = 'local.zip'
    var_1 = 'local_project/'
    var_2 = 'local_project/file.txt'
    var_3 = False
    var_4 = True
    var_5 = 'local_project'

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com/empty.zip'
    var_2 = True
    var_3 = True
    var_4 = bool(var_3)
    assert var_4 is True

def test_case_0():
    var_0 = 'file.txt'
    var_1 = False
    var_2 = 'http://example.com/bad.zip'
    var_3 = True
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 'protected_project/'
    var_1 = 'protected_project/file.txt'
    var_2 = []
    var_3 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_unzip_with_url_creates_clone_dir. Retrieved 7/28 statements.
# Partially parsed test_unzip_local_file. Retrieved 6/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/16 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 6/22 statements.
# Partially parsed test_unzip_with_url_prompt_and_delete. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'new_clone_dir'
    var_1 = b'chunk1'
    var_2 = b'chunk2'
    var_3 = 'project_dir/'
    var_4 = 'project_dir/file.txt'
    var_5 = 'http://example.com/repo.zip'
    var_6 = True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_dir/'
    var_1 = 'project_dir/file.txt'
    var_2 = '/local/path/repo.zip'
    var_3 = False
    var_4 = '.'
    var_5 = module_0.unzip(var_2, var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = '/local/path/repo.zip'
    var_2 = False
    var_3 = module_0.unzip(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_dir/'
    var_1 = 'project_dir/file.txt'
    var_2 = '/local/path/repo.zip'
    var_3 = False
    var_4 = 'mypassword'
    var_5 = module_0.unzip(var_2, var_3, password=var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'clone'
    var_1 = 'repo.zip'
    var_2 = b'chunk1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_unzip_predicate_line_54_evaluates_to_false. Retrieved 9/28 statements.


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
    var_9 = bool(var_5)
    assert var_9 is True



