####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/21 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 5/15 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 7/17 statements.
# Partially parsed test_unzip_invalid_zipfile. Retrieved 6/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip with an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'Test unzip with zipfile missing top-level directory raises InvalidZipRepository.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip with invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True

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
    var_10 = True
    var_11 = module_0.exists()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/20 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 11/22 statements.
# Partially parsed test_unzip_with_wrong_password_raises_error. Retrieved 12/25 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notopdir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False

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
    var_10 = module_0.exists()

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and password provided.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'correct_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False
    var_11 = 'wrong_password'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = 'clone'
    var_10 = False
    var_11 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_on_bad_zipfile. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_predicate_at_line_36_evaluates_to_false. Retrieved 14/30 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when is_url is True, the zip file exists, and prompt_and_delete\n    returns False (user chooses to reuse existing version).\n    '
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'repo.zip'
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = False
    var_5 = module_0.patch(var_3)
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = module_0.patch(var_6)
    var_8 = 'project_dir/'
    var_9 = 'project_dir/file.txt'
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_12 = 'temp'
    var_13 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true_when_zip_path_exists. Retrieved 13/41 statements.


import _io as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'test.zip'
    var_2 = []
    var_3 = module_0.BytesIO()
    var_4 = 'test_project/'
    var_5 = ''
    var_6 = 'test_project/file.txt'
    var_7 = 'content'
    var_8 = 0
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = 'cookiecutter.zipfile.prompt_and_delete'
    var_11 = 'clone'
    var_12 = 'test.zip'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 10/30 statements.
# Partially parsed test_unzip_creates_clone_dir. Retrieved 11/28 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/11 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 10/25 statements.


def test_case_0():
    var_0 = 'zip_source'
    var_1 = 'test_project'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'test.zip'
    var_5 = 'test_project/'
    var_6 = 'file.txt'
    var_7 = 'test_project/file.txt'
    var_8 = 'clone'
    var_9 = False

import genericpath as module_0

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'test.zip'
    var_4 = 'test_project/'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'nonexistent'
    var_8 = 'clone'
    var_9 = False
    var_10 = module_0.exists()

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'content'
    var_2 = 'no_dir.zip'
    var_3 = 'file.txt'
    var_4 = 'clone'
    var_5 = False

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = b'This is not a zip file'
    var_2 = 'clone'
    var_3 = False

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'test.zip'
    var_4 = 'test_project/'
    var_5 = 'file.txt'
    var_6 = 'test_project/file.txt'
    var_7 = 'clone'
    var_8 = False
    var_9 = '~'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 9/23 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/23 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/11 statements.
# Partially parsed test_unzip_url_no_input_downloads. Retrieved 9/23 statements.
# Partially parsed test_unzip_url_existing_file_no_input. Retrieved 11/26 statements.
# Partially parsed test_unzip_with_password. Retrieved 11/21 statements.
# Partially parsed test_unzip_expanduser_in_clone_to_dir. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test unzipping a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = 'project_name'

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'nonexistent'
    var_3 = 'dir'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test that unzip raises error when zip has no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True downloads the file.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'http://example.com/test.zip'
    var_7 = True
    var_8 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = [var_6]
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'project_name'

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False
    var_10 = 'project_name'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that unzip expands ~ in clone_to_dir path.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = '~/test'
    var_8 = module_0.unzip(var_2, var_6, var_7)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for empty chunks.'
    var_1 = b''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 18/42 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'test.zip'
    var_2 = b'mock'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'cookiecutter.zipfile.prompt_and_delete'
    var_7 = False
    var_8 = lambda path, no_input=False: var_7
    var_9 = 'os.path.exists'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = 'project_name/'
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = 'os.path.abspath'
    var_15 = 'tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = 'http://example.com/test.zip'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_creates_clone_to_dir. Retrieved 21/36 statements.
# Partially parsed test_unzip_local_file_without_url. Retrieved 12/24 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 18/30 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 19/31 statements.
# Partially parsed test_unzip_bad_zipfile_raises_error. Retrieved 19/30 statements.
# Partially parsed test_unzip_with_password_protection_and_valid_password. Retrieved 8/10 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'new_dir'
    var_2 = 'http://example.com/test.zip'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = 'cookiecutter.zipfile.prompt_and_delete'
    var_6 = True
    var_7 = module_0.patch(var_5)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = module_0.patch(var_10)
    var_12 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_13 = 'temp'
    var_14 = b'test_chunk'
    var_15 = 'project_name/'
    var_16 = 'project_name/file.txt'
    var_17 = 'builtins.open'
    var_18 = 'cookiecutter.zipfile.os.path.exists'
    var_19 = False
    var_20 = module_0.patch(var_18)

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip works with local file path.'
    var_1 = 'test.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'cookiecutter.zipfile.ZipFile'
    var_5 = module_0.patch(var_4)
    var_6 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_7 = 'temp'
    var_8 = 'project_name/'
    var_9 = 'project_name/file.txt'
    var_10 = False
    var_11 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = True
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = False
    var_12 = module_0.patch(var_10)
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = module_0.patch(var_13)
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = True
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = False
    var_12 = module_0.patch(var_10)
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = module_0.patch(var_13)
    var_15 = 'file.txt'
    var_16 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_17 = 'temp'
    var_18 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = True
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = 'builtins.open'
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = False
    var_12 = module_0.patch(var_10)
    var_13 = 'cookiecutter.zipfile.ZipFile'
    var_14 = module_0.patch(var_13)
    var_15 = 'Bad zip'
    var_16 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_17 = 'temp'
    var_18 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and valid password.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'correct_password'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = 'cookiecutter.zipfile.prompt_and_delete'
    var_6 = True
    var_7 = module_0.patch(var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_with_url_and_new_download. Retrieved 18/30 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 13/23 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 16/29 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 17/30 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 17/29 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/17 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist yet."
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = module_0.patch(var_3)
    var_5 = b'PK\x03\x04'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = module_0.patch(var_6)
    var_8 = 'project_name/'
    var_9 = 'project_name/file.txt'
    var_10 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_11 = module_0.patch(var_10)
    var_12 = 'cookiecutter.zipfile.os.path.exists'
    var_13 = False
    var_14 = module_0.patch(var_12)
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = 'repo.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'cookiecutter.zipfile.ZipFile'
    var_4 = module_0.patch(var_3)
    var_5 = 'project_name/'
    var_6 = 'project_name/file.txt'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_10 = 'temp'
    var_11 = False
    var_12 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that empty zip file raises InvalidZipRepository.'
    var_1 = 'https://example.com/empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = module_0.patch(var_3)
    var_5 = b'PK\x03\x04'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = module_0.patch(var_6)
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = False
    var_12 = module_0.patch(var_10)
    var_13 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_14 = 'temp'
    var_15 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that zip without top-level directory raises InvalidZipRepository.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = module_0.patch(var_3)
    var_5 = b'PK\x03\x04'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = module_0.patch(var_6)
    var_8 = 'file.txt'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = module_0.patch(var_9)
    var_11 = 'cookiecutter.zipfile.os.path.exists'
    var_12 = False
    var_13 = module_0.patch(var_11)
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = 'temp'
    var_16 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that bad zip file raises InvalidZipRepository.'
    var_1 = 'https://example.com/bad.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = module_0.patch(var_3)
    var_5 = b'invalid'
    var_6 = 'cookiecutter.zipfile.ZipFile'
    var_7 = module_0.patch(var_6)
    var_8 = 'Bad zip'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = module_0.patch(var_9)
    var_11 = 'cookiecutter.zipfile.os.path.exists'
    var_12 = False
    var_13 = module_0.patch(var_11)
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = 'temp'
    var_16 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'https://example.com/protected.zip'
    var_2 = 'clone'
    var_3 = 'test_password'
    var_4 = 'cookiecutter.zipfile.requests.get'
    var_5 = module_0.patch(var_4)
    var_6 = b'PK\x03\x04'
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = module_0.patch(var_7)
    var_9 = 'project_name/'
    var_10 = 'project_name/file.txt'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_raises_error_when_zip_is_empty. Retrieved 6/16 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = False
    var_5 = module_1.unzip(var_0, var_4, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_download_predicate_false. Retrieved 13/36 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.prompt_and_delete'
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'http://example.com/test.zip'
    var_10 = True
    var_11 = False
    var_12 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_line_39_predicate_evaluates_to_true. Retrieved 16/37 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = b'chunk1'
    var_7 = b'chunk2'
    var_8 = b''
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = 'test_dir/'
    var_11 = 'test_dir/file.txt'
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = 1024



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_with_url_creates_clone_to_dir. Retrieved 13/18 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 11/23 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 11/19 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 12/20 statements.
# Partially parsed test_unzip_bad_zipfile_raises_error. Retrieved 12/18 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 18/30 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 8/13 statements.


import requests.api as module_0

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'clone'
    var_2 = 'http://example.com/repo.zip'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = 'cookiecutter.zipfile.os.path.exists'
    var_6 = False
    var_7 = module_0.patch(var_5)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = module_0.patch(var_10)
    var_12 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = False
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_9 = 'temp'
    var_10 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'http://example.com/empty.zip'
    var_2 = False
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = 'cookiecutter.zipfile.os.path.exists'
    var_6 = module_0.patch(var_5)
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises error when zip has no top-level directory.'
    var_1 = 'http://example.com/notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = False
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = 'cookiecutter.zipfile.os.path.exists'
    var_7 = module_0.patch(var_6)
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for bad zipfile.'
    var_1 = 'http://example.com/bad.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = 'cookiecutter.zipfile.os.path.exists'
    var_5 = False
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.ZipFile'
    var_10 = 'Bad zip'
    var_11 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected file and valid password.'
    var_1 = 'http://example.com/protected.zip'
    var_2 = 'correct_password'
    var_3 = 'project/'
    var_4 = 'project/file.txt'
    var_5 = 'Bad password'
    var_6 = None
    var_7 = False
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.os.path.exists'
    var_11 = module_0.patch(var_10)
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = module_0.patch(var_12)
    var_14 = 'cookiecutter.zipfile.ZipFile'
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'temp'
    var_17 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected file and no_input=True raises error.'
    var_1 = 'http://example.com/protected.zip'
    var_2 = 'project/'
    var_3 = 'project/file.txt'
    var_4 = 'Bad password'
    var_5 = False
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = module_0.patch(var_6)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 19/40 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/25 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 6/19 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/21 statements.
# Partially parsed test_unzip_invalid_zipfile_raises_error. Retrieved 5/13 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 11/25 statements.
# Partially parsed test_unzip_url_not_exists_downloads. Retrieved 13/31 statements.


import _io as module_0
import requests.api as module_1
import genericpath as module_2

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a URL-based zipfile.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clones'
    var_3 = module_0.BytesIO()
    var_4 = 'project-name/'
    var_5 = ''
    var_6 = 'project-name/file.txt'
    var_7 = 'content'
    var_8 = 0
    var_9 = 'requests.get'
    var_10 = 'cookiecutter.zipfile.prompt_and_delete'
    var_11 = True
    var_12 = module_1.patch(var_10)
    var_13 = 'temp'
    var_14 = 'tempfile.mkdtemp'
    var_15 = False
    var_16 = 'project-name'
    var_17 = 'repo.zip'
    var_18 = module_2.exists()

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile path.'
    var_1 = module_0.BytesIO()
    var_2 = 'local-project/'
    var_3 = ''
    var_4 = 'local-project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'local.zip'
    var_8 = 'clones'
    var_9 = False

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = module_0.BytesIO()
    var_2 = 0
    var_3 = 'empty.zip'
    var_4 = 'clones'
    var_5 = False

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when zipfile lacks top-level directory.'
    var_1 = module_0.BytesIO()
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 0
    var_5 = 'notoplevel.zip'
    var_6 = 'clones'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a valid zip file'
    var_3 = 'clones'
    var_4 = False

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile and valid password.'
    var_1 = module_0.BytesIO()
    var_2 = 'protected-project/'
    var_3 = ''
    var_4 = 'protected-project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'protected.zip'
    var_8 = 'clones'
    var_9 = False
    var_10 = 'test'

import _io as module_0

def test_case_0():
    var_0 = "Test unzip downloads file when URL-based zipfile doesn't exist locally."
    var_1 = 'https://example.com/newrepo.zip'
    var_2 = 'clones'
    var_3 = module_0.BytesIO()
    var_4 = 'newproject/'
    var_5 = ''
    var_6 = 'newproject/file.txt'
    var_7 = 'content'
    var_8 = 0
    var_9 = 'requests.get'
    var_10 = 'temp'
    var_11 = 'tempfile.mkdtemp'
    var_12 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 17/44 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'chunk1'
    var_2 = b''
    var_3 = b'chunk2'
    var_4 = b'chunk3'
    var_5 = [var_1, var_2, var_3, var_2, var_4]
    var_6 = 'test.zip'
    var_7 = 'test_dir/'
    var_8 = ''
    var_9 = 'test_dir/file.txt'
    var_10 = 'content'
    var_11 = []
    var_12 = 'test_dir/'
    var_13 = 'test_dir/file.txt'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = len(var_11)
    assert var_16 == 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 9/33 statements.
# Partially parsed test_unzip_with_local_file_extracts. Retrieved 6/21 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 1/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 3/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 2/13 statements.
# Partially parsed test_unzip_with_password_protected_zip_with_correct_password. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'test_project/'
    var_1 = ''
    var_2 = 'test_project/file.txt'
    var_3 = 'content'
    var_4 = 'rb'
    var_5 = 'http://example.com/test.zip'
    var_6 = True
    var_7 = False
    var_8 = 'test_project'

def test_case_0():
    var_0 = 'local_project/'
    var_1 = ''
    var_2 = 'local_project/file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = 'local_project'

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'content'
    var_2 = False

def test_case_0():
    var_0 = b'not a valid zip file'
    var_1 = False

def test_case_0():
    var_0 = 'pwd_project/'
    var_1 = ''
    var_2 = 'pwd_project/file.txt'
    var_3 = 'content'
    var_4 = b'test_password'
    var_5 = False
    var_6 = 'test_password'
    var_7 = 'pwd_project'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 8/31 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/23 statements.
# Partially parsed test_unzip_empty_repository_raises_error. Retrieved 2/15 statements.
# Partially parsed test_unzip_missing_top_level_directory_raises_error. Retrieved 3/16 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 8/24 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 2/13 statements.


def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = [var_0, var_1]
    var_3 = 'project_name/'
    var_4 = 'project_name/file.txt'
    var_5 = 'https://example.com/repo.zip'
    var_6 = True
    var_7 = 'project_name'

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = 'local.zip'
    var_3 = False
    var_4 = True
    var_5 = 'project_name'

def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = True

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'https://example.com/repo.zip'
    var_2 = True

import builtins as module_0

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = module_0.RuntimeError()
    var_3 = None
    var_4 = 'https://example.com/repo.zip'
    var_5 = True
    var_6 = 'mypassword'
    var_7 = 'project_name'

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = 'http://example.com/test.zip'
    var_4 = True
    var_5 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/21 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 6/15 statements.
# Partially parsed test_unzip_url_with_no_input_downloads. Retrieved 12/31 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 10/22 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 14/27 statements.
# Partially parsed test_unzip_clone_to_dir_created_if_not_exists. Retrieved 12/21 statements.
# Partially parsed test_unzip_expanduser_in_clone_to_dir. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_top_dir.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = module_0.unzip(var_3, var_5, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = 'This is not a zip file'
    var_4 = False
    var_5 = module_0.unzip(var_0, var_4, var_2)

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True.'
    var_1 = 'temp_zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'test.zip'
    var_8 = 'rb'
    var_9 = 'requests.get'
    var_10 = 'http://example.com/test.zip'
    var_11 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = b'test_password'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = 'test_password'

import zipfile as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zip, no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = module_0.ZipInfo(var_5)
    var_7 = var_6.flag_bits
    var_8 = 1
    var_9 = var_7 | var_8
    var_10 = 'content'
    var_11 = False
    var_12 = True
    var_13 = module_1.unzip(var_3, var_11, var_2, var_12)

import locale as module_0
import genericpath as module_1

def test_case_0():
    var_0 = "Test that clone_to_dir is created if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'nonexistent'
    var_7 = 'clone'
    var_8 = var_4 / var_7
    var_9 = False
    var_10 = module_0.str(var_8)
    var_11 = module_1.exists(var_8)

def test_case_0():
    var_0 = 'Test that clone_to_dir expands user home directory.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 7/15 statements.
# Partially parsed test_unzip_with_url_no_input. Retrieved 11/26 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 10/21 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 10/23 statements.


def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = False
    var_4 = True
    var_5 = module_0.unzip(var_0, var_3, var_2, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = True
    var_7 = module_0.unzip(var_3, var_5, var_2, var_6)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'extract'
    var_4 = False
    var_5 = True
    var_6 = module_0.unzip(var_0, var_4, var_2, var_5)

def test_case_0():
    var_0 = 'Test unzip with URL and no_input=True.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'rb'
    var_8 = 'requests.get'
    var_9 = 'http://example.com/test.zip'
    var_10 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and provided password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = True

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'nonexistent'
    var_3 = 'path'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = False
    var_9 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 14/28 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This happens when is_url is True, the zip file exists, and prompt_and_delete\n    returns False (user chooses to reuse existing version).\n    '
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'clone'
    var_3 = 'repo.zip'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = False
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = module_0.patch(var_7)
    var_9 = 'project_name/'
    var_10 = True
    var_11 = None
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = module_0.patch(var_12)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts_zipfile. Retrieved 13/38 statements.
# Partially parsed test_unzip_with_local_file_extracts_zipfile. Retrieved 11/27 statements.
# Partially parsed test_unzip_raises_on_empty_zipfile. Retrieved 5/15 statements.
# Partially parsed test_unzip_raises_on_missing_top_level_directory. Retrieved 7/17 statements.
# Partially parsed test_unzip_raises_on_invalid_zipfile. Retrieved 6/14 statements.
# Partially parsed test_unzip_with_password_protected_zipfile. Retrieved 13/25 statements.
# Partially parsed test_unzip_creates_clone_directory_if_not_exists. Retrieved 12/23 statements.
# Partially parsed test_unzip_with_url_skips_download_if_not_needed. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'Test unzip downloads a URL-based zipfile and extracts it.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = 'cookiecutter.zipfile.prompt_and_delete'
    var_9 = True
    var_10 = lambda path, no_input: var_9
    var_11 = 'http://example.com/test.zip'
    var_12 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip with local file path extracts the zipfile.'
    var_1 = 'local.zip'
    var_2 = 'myproject/'
    var_3 = ''
    var_4 = 'myproject/test.txt'
    var_5 = 'test content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'myproject'
    var_10 = 'test.txt'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository if top-level is not a directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zipfile.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zipfile'
    var_3 = 'clone'
    var_4 = False
    var_5 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'utf-8'
    var_4 = module_0.encode(var_3)
    var_5 = 'project/'
    var_6 = ''
    var_7 = 'project/file.txt'
    var_8 = 'content'
    var_9 = 'clone'
    var_10 = False
    var_11 = True
    var_12 = 'project'

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
    var_10 = True
    var_11 = module_0.exists()

def test_case_0():
    var_0 = 'Test unzip with URL skips download if file exists and user chooses to reuse.'
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.prompt_and_delete'
    var_8 = False
    var_9 = lambda path, no_input: var_8
    var_10 = 'http://example.com/test.zip'
    var_11 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_predicate_line_39_evaluates_to_false. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to False.'
    var_1 = b''
    var_2 = None
    var_3 = [var_1, var_2, var_1]
    var_4 = 'http://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'project_dir/'
    var_7 = True
    var_8 = 1024



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 12/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 7/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 10/21 statements.
# Partially parsed test_unzip_with_valid_password. Retrieved 12/21 statements.
# Partially parsed test_unzip_expanduser_clone_to_dir. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = False
    var_11 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = True

def test_case_0():
    var_0 = 'Test unzip with password protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = b'password'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = False
    var_9 = True

def test_case_0():
    var_0 = 'Test unzip with password protected zip and valid password.'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = False
    var_10 = True
    var_11 = 'test'

def test_case_0():
    var_0 = 'Test unzip expands user path for clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'proj/'
    var_3 = ''
    var_4 = 'proj/file.txt'
    var_5 = 'content'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = False
    var_10 = '.'
    var_11 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_bad_zip_file_raises_invalid_zip_repository. Retrieved 8/18 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 12/27 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 7/16 statements.
# Partially parsed test_unzip_url_with_existing_file_no_input. Retrieved 17/32 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 19/38 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 12/24 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_11 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = False

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
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.os.path.exists'
    var_10 = True
    var_11 = module_0.patch(var_9)
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = False
    var_14 = module_0.patch(var_12)
    var_15 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_16 = 'extract'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile using provided password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'extract'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = module_0.patch(var_9)
    var_11 = 'cookiecutter.zipfile.ZipFile'
    var_12 = module_0.patch(var_11)
    var_13 = 'project_name/'
    var_14 = 'project_name/file.txt'
    var_15 = 'Bad password'
    var_16 = None
    var_17 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_18 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile using invalid password.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'clone'
    var_5 = 'extract'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = module_0.patch(var_6)
    var_8 = 'cookiecutter.zipfile.ZipFile'
    var_9 = module_0.patch(var_8)
    var_10 = 'project_name/'
    var_11 = 'project_name/file.txt'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 11/24 statements.
# Partially parsed test_unzip_url_no_existing_file. Retrieved 16/31 statements.
# Partially parsed test_unzip_url_with_existing_file_delete. Retrieved 17/32 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 6/17 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 8/19 statements.
# Partially parsed test_unzip_invalid_zip_file. Retrieved 7/17 statements.
# Partially parsed test_unzip_password_protected_with_password. Retrieved 17/32 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = False
    var_10 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a URL when no cached file exists.'
    var_1 = 'temp.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'os.path.exists'
    var_10 = False
    var_11 = module_0.patch(var_9)
    var_12 = 'requests.get'
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a URL when cached file exists and user chooses to delete.'
    var_1 = 'temp.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'os.path.exists'
    var_10 = True
    var_11 = module_0.patch(var_9)
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = module_0.patch(var_12)
    var_14 = 'requests.get'
    var_15 = 'http://example.com/test.zip'
    var_16 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping an empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a zipfile without top-level directory raises InvalidZipRepository.'
    var_1 = 'no_dir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping an invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = False

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzipping a password-protected zipfile with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'testpass'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = 'project_name/'
    var_11 = 'project_name/file.txt'
    var_12 = 'Bad password'
    var_13 = None
    var_14 = 'cookiecutter.zipfile.ZipFile'
    var_15 = 'tempfile.mkdtemp'
    var_16 = 'temp'



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 41 (if chunk:) evaluates to False for empty chunks.'
    var_1 = b''
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_predicate_line_31_true. Retrieved 8/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to True when zip_path exists.'
    var_1 = 'test.zip'
    var_2 = b'PK\x03\x04'
    var_3 = 'clone'
    var_4 = 'project/'
    var_5 = 'http://example.com/test.zip'
    var_6 = True
    var_7 = False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_zipfile_predicate_line_54_evaluates_to_false. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This ensures that when a zipfile has at least one entry, the empty check passes.\n    '
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 15/34 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This occurs when prompt_and_delete returns False, indicating the user\n    wants to reuse the existing zipfile instead of re-downloading it.\n    '
    var_1 = 'clone'
    var_2 = 'https://example.com/repo.zip'
    var_3 = 'repo.zip'
    var_4 = 'cookiecutter.zipfile.prompt_and_delete'
    var_5 = False
    var_6 = module_0.patch(var_4)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = module_0.patch(var_7)
    var_9 = 'project_name/'
    var_10 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_11 = 'temp'
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = module_0.patch(var_12)
    var_14 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/20 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 11/20 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 9/21 statements.
# Partially parsed test_unzip_with_url_no_input. Retrieved 10/26 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False

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
    var_10 = module_0.exists()

def test_case_0():
    var_0 = 'Test unzip raises error for password protected zip with no_input=True.'
    var_1 = 'protected.zip'
    var_2 = b'password'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = True

def test_case_0():
    var_0 = 'Test unzip succeeds with correct password provided.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = 'password'

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist yet."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'rb'
    var_8 = 'http://example.com/test.zip'
    var_9 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 14/21 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.'
    var_1 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_2 = module_0.patch(var_1)
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = False
    var_5 = module_0.patch(var_3)
    var_6 = 'cookiecutter.zipfile.os.path.exists'
    var_7 = True
    var_8 = module_0.patch(var_6)
    var_9 = 'project_dir/'
    var_10 = 'cookiecutter.zipfile.ZipFile'
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = module_0.patch(var_11)
    var_13 = 'https://example.com/repo.zip'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_iter_content_filters_empty_chunks. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for empty chunks.'
    var_1 = b'data'
    var_2 = b''
    var_3 = b'more'
    var_4 = [var_1, var_2, var_3, var_2]
    var_5 = 'test.zip'
    var_6 = 'test_dir/'
    var_7 = ''
    var_8 = 'test_dir/file.txt'
    var_9 = 'content'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_context_manager_with_zipfile. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 evaluates to True by verifying ZipFile context manager is used.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 16/29 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/19 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 8/18 statements.
# Partially parsed test_unzip_with_password_protection. Retrieved 13/23 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 7/15 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 17/30 statements.
# Partially parsed test_unzip_with_password_wrong_password_raises_error. Retrieved 12/24 statements.


import _io as module_0
import requests.api as module_1

def test_case_0():
    var_0 = "Test unzip with a URL when the file doesn't exist yet."
    var_1 = module_0.BytesIO()
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = [var_4]
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = module_1.patch(var_9)
    var_11 = 'cookiecutter.zipfile.os.path.exists'
    var_12 = False
    var_13 = module_1.patch(var_11)
    var_14 = 'http://example.com/test.zip'
    var_15 = True

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'test.zip'
    var_2 = 'local_project/'
    var_3 = ''
    var_4 = 'local_project/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_7 = module_0.patch(var_6)
    var_8 = False

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = False
    var_5 = module_1.unzip(var_0, var_4, var_2)

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_topdir.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = False
    var_7 = module_1.unzip(var_2, var_6, var_4)

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip file.'
    var_1 = 'protected.zip'
    var_2 = b'test_password'
    var_3 = 'protected_project/'
    var_4 = ''
    var_5 = 'protected_project/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.os.path.exists'
    var_10 = False
    var_11 = module_0.patch(var_9)
    var_12 = 'test_password'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for corrupted zip.'
    var_1 = 'bad.zip'
    var_2 = 'not a zip file'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False
    var_6 = module_1.unzip(var_0, var_5, var_2)

import _io as module_0
import requests.api as module_1

def test_case_0():
    var_0 = 'Test unzip with URL and existing file with no_input=True.'
    var_1 = module_0.BytesIO()
    var_2 = 'url_project/'
    var_3 = ''
    var_4 = 'url_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = [var_4]
    var_8 = 'cookiecutter.zipfile.requests.get'
    var_9 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_10 = module_1.patch(var_9)
    var_11 = 'cookiecutter.zipfile.os.path.exists'
    var_12 = True
    var_13 = module_1.patch(var_11)
    var_14 = 'cookiecutter.zipfile.rmtree'
    var_15 = module_1.patch(var_14)
    var_16 = 'http://example.com/test.zip'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository with wrong password.'
    var_1 = 'protected.zip'
    var_2 = b'correct_password'
    var_3 = 'protected_project/'
    var_4 = ''
    var_5 = 'protected_project/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = False
    var_10 = 'wrong_password'
    var_11 = module_1.unzip(var_2, var_9, var_7, password=var_10)

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for password-protected with no_input.'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 15/40 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = b'chunk1'
    var_2 = b'chunk2'
    var_3 = None
    var_4 = b'chunk3'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = 'test.zip'
    var_7 = 'test_dir/'
    var_8 = ''
    var_9 = 'test_dir/file.txt'
    var_10 = 'content'
    assert var_10 == 3
    var_11 = 'test_dir/'
    var_12 = 'test_dir/file.txt'
    var_13 = 'http://example.com/test.zip'
    var_14 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 7/21 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 55 evaluates to True when zip is empty.'
    var_1 = 'empty.zip'
    var_2 = 'make_sure_path_exists'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = module_0.unzip(var_0, var_5, var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 10/26 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/16 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'extract'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = 'clone'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'notoplevel.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'clone'
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = 'clone'
    var_3 = False
    var_4 = True

import genericpath as module_0

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
    var_9 = module_0.exists()
    var_10 = 'project_name'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 2/11 statements.
# Partially parsed test_unzip_missing_top_level_directory_raises_error. Retrieved 4/13 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 3/10 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/20 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 9/19 statements.
# Partially parsed test_unzip_with_wrong_password_raises_error. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'extract'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'test_project'

def test_case_0():
    var_0 = 'empty.zip'
    var_1 = False

def test_case_0():
    var_0 = 'no_toplevel.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False

def test_case_0():
    var_0 = 'invalid.zip'
    var_1 = 'not a zip file'
    var_2 = False

import genericpath as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project/'
    var_2 = ''
    var_3 = 'project/file.txt'
    var_4 = 'content'
    var_5 = 'new_clone_dir'
    var_6 = False
    var_7 = module_0.exists()
    var_8 = 'project'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = b'password'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = b'mypassword'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'mypassword'
    var_8 = 'project'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = b'correctpassword'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'wrongpassword'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_unzip_opens_zipfile_with_context_manager. Retrieved 10/29 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 evaluates to True - ZipFile is opened.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/11 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 8/20 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input. Retrieved 10/20 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/20 statements.
# Partially parsed test_unzip_with_wrong_password_raises_error. Retrieved 9/20 statements.
# Partially parsed test_unzip_expanduser_in_clone_to_dir. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = False

def test_case_0():
    var_0 = 'Test unzip with empty zipfile raises InvalidZipRepository.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip without top-level directory raises InvalidZipRepository.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with invalid zipfile raises InvalidZipRepository.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = False
    var_4 = module_0.unzip(var_0, var_3, var_2)

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'new_clone_dir'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and no_input=True raises error.'
    var_1 = 'protected.zip'
    var_2 = b'test'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = False
    var_8 = True
    var_9 = module_0.unzip(var_2, var_7, var_3, var_8)

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'utf-8'
    var_8 = module_0.encode(var_7)
    var_9 = False

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = b'correct_password'
    var_7 = False
    var_8 = 'wrong_password'

def test_case_0():
    var_0 = 'Test unzip expands user home directory in clone_to_dir.'
    var_1 = 'test.zip'
    var_2 = 'project/'
    var_3 = ''
    var_4 = 'project/file.txt'
    var_5 = 'content'
    var_6 = 'HOME'
    var_7 = '~/cookiecutter'
    var_8 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 10/25 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 12/22 statements.
# Partially parsed test_unzip_with_url_and_no_input. Retrieved 12/35 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 10/21 statements.


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
    var_9 = 'project_name'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = 'bad.zip'
    var_2 = 'This is not a zip file'
    var_3 = 'clone'
    var_4 = False

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
    var_10 = module_0.exists()
    var_11 = 'project_name'

import genericpath as module_0

def test_case_0():
    var_0 = "Test unzip with URL when file doesn't exist yet."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = 'get'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'project_name'
    var_11 = module_0.exists()

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile.'
    var_1 = 'protected.zip'
    var_2 = 'secret'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = False
    var_9 = 'project_name'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_unzip_writes_chunk_to_file. Retrieved 17/32 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to True when chunk is not empty.'
    var_1 = 'test.zip'
    var_2 = 'clone'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = b'test_chunk_data'
    var_8 = None
    var_9 = b'more_data'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = True
    var_14 = module_0.patch(var_12)
    var_15 = 'http://example.com/test.zip'
    var_16 = 1024



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 7/24 statements.
# Partially parsed test_unzip_with_local_file_extracts. Retrieved 5/18 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_with_password_protected_file. Retrieved 6/20 statements.
# Partially parsed test_unzip_invalid_zip_file_raises_error. Retrieved 2/11 statements.
# Partially parsed test_unzip_url_file_exists_with_no_input. Retrieved 5/19 statements.


def test_case_0():
    var_0 = b'test_data'
    var_1 = [var_0]
    var_2 = 'project_name/'
    var_3 = 'project_name/file.txt'
    var_4 = [var_2, var_3]
    var_5 = 'http://example.com/repo.zip'
    var_6 = True

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/local.zip'
    var_4 = False

def test_case_0():
    var_0 = []
    var_1 = '/path/to/local.zip'
    var_2 = False

def test_case_0():
    var_0 = 'file.txt'
    var_1 = [var_0]
    var_2 = '/path/to/local.zip'
    var_3 = False

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = [var_0, var_1]
    var_3 = '/path/to/local.zip'
    var_4 = False
    var_5 = 'test_password'

def test_case_0():
    var_0 = '/path/to/invalid.zip'
    var_1 = False

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = [var_0, var_1]
    var_3 = 'http://example.com/repo.zip'
    var_4 = True



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unzip_local_file. Retrieved 8/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 2/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_url_with_no_input_and_no_existing_file. Retrieved 11/26 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 11/21 statements.
# Partially parsed test_unzip_password_protected_no_input_raises_error. Retrieved 8/19 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 8/21 statements.
# Partially parsed test_unzip_with_expanduser. Retrieved 10/21 statements.


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
    var_0 = 'empty.zip'
    var_1 = False

def test_case_0():
    var_0 = 'no_top_level.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = False

def test_case_0():
    var_0 = 'bad.zip'
    var_1 = 'this is not a zip file'
    var_2 = False

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'clone'
    var_1 = 'source.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = [var_4]
    var_7 = 'http://example.com/project.zip'
    var_8 = True
    var_9 = module_0.unzip(var_7, var_8, var_1, var_8)
    var_10 = 'project_name'

import email._encoded_words as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'secret'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = False
    var_9 = module_1.unzip(var_3, var_8, var_4, password=var_1)
    var_10 = 'project_name'

def test_case_0():
    var_0 = 'protected.zip'
    var_1 = 'secret'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True

def test_case_0():
    var_0 = 'new_dir'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = 'project_name'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'HOME'
    var_6 = False
    var_7 = '~/clone'
    var_8 = module_0.unzip(var_4, var_6, var_7)
    var_9 = 'project_name'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_unzip_download_predicate_false. Retrieved 14/29 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when prompt_and_delete returns False.'
    var_1 = 'http://example.com/test.zip'
    var_2 = 'test.zip'
    var_3 = 'cookiecutter.zipfile.prompt_and_delete'
    var_4 = False
    var_5 = module_0.patch(var_3)
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = module_0.patch(var_6)
    var_8 = 'cookiecutter.zipfile.ZipFile'
    var_9 = module_0.patch(var_8)
    var_10 = 'project/'
    var_11 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_12 = 'temp'
    var_13 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_bad_zipfile_exception_handling. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and converted to InvalidZipRepository.'
    var_1 = 'bad.zip'
    var_2 = b'This is not a valid zip file'
    var_3 = False
    var_4 = True



# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_predicate_line_54_evaluates_to_false. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 54 (len(zip_file.namelist()) == 0) evaluates to False.\n    \n    This means the zipfile contains at least one entry.\n    '
    var_1 = 'test.zip'
    var_2 = 'test_dir/'
    var_3 = ''
    var_4 = 'test_dir/file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter.zipfile.requests.get'
    var_7 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_downloads_zipfile_when_download_is_true. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if download:) evaluates to True.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = b'test chunk'
    var_3 = [var_2]
    var_4 = 'project-name/'
    var_5 = [var_4]
    var_6 = None
    var_7 = True
    var_8 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_unzip_with_url_new_download. Retrieved 10/29 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/27 statements.
# Partially parsed test_unzip_empty_zipfile_raises_error. Retrieved 6/19 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/20 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_password_protected_with_provided_password. Retrieved 13/32 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'Test unzip downloads a new zipfile from URL.'
    var_1 = 'https://example.com/repo.zip'
    var_2 = 'cloned'
    var_3 = b'PK\x03\x04'
    var_4 = [var_3]
    var_5 = 'project/'
    var_6 = False
    var_7 = True
    var_8 = 'temp'
    var_9 = 'project'

def test_case_0():
    var_0 = 'Test unzip with a local zipfile path.'
    var_1 = 'local.zip'
    var_2 = 'cloned'
    var_3 = 'myproject/'
    var_4 = False
    var_5 = False
    var_6 = True
    var_7 = 'temp'
    var_8 = 'myproject'

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = 'empty.zip'
    var_2 = 'cloned'
    var_3 = False
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'Test unzip raises error when zip has no top-level directory.'
    var_1 = 'bad.zip'
    var_2 = 'cloned'
    var_3 = 'file.txt'
    var_4 = False
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zip archive.'
    var_1 = 'invalid.zip'
    var_2 = 'cloned'
    var_3 = False
    var_4 = True

import locale as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected archive and provided password.'
    var_1 = 'protected.zip'
    var_2 = 'cloned'
    var_3 = 'secret'
    var_4 = 'project/'
    var_5 = 'Bad password'
    var_6 = None
    var_7 = False
    var_8 = False
    var_9 = True
    var_10 = 'temp'
    var_11 = module_0.str(var_7)
    var_12 = 'project'

def test_case_0():
    var_0 = 'Test unzip with invalid password for protected archive.'
    var_1 = 'protected.zip'
    var_2 = 'cloned'
    var_3 = 'wrong'
    var_4 = 'project/'
    var_5 = 'Bad password'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 9/20 statements.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_predicate_line_36_evaluates_to_false. Retrieved 18/48 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 (if download:) evaluates to False.\n    \n    This happens when is_url is True, zip_path exists, and prompt_and_delete\n    returns False (user chooses to reuse existing version).\n    '
    var_1 = 'test.zip'
    var_2 = b'fake zip content'
    var_3 = 'clone'
    var_4 = 'test.zip'
    var_5 = 'cookiecutter.zipfile.prompt_and_delete'
    var_6 = 'get'
    var_7 = 0
    var_8 = {var_6: var_7}
    var_9 = 'cookiecutter.zipfile.requests.get'
    var_10 = 'project_name/'
    var_11 = False
    var_12 = 'cookiecutter.zipfile.ZipFile'
    var_13 = 'temp'
    var_14 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_15 = 'https://example.com/test.zip'
    var_16 = True
    var_17 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_unzip_with_valid_zipfile. Retrieved 13/34 statements.


import requests.cookies as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 54 evaluates to True with a valid zipfile.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = 'test_project/'
    var_4 = ''
    var_5 = 'test_project/file.txt'
    var_6 = 'content'
    var_7 = 'get'
    var_8 = module_0.MockResponse()
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 13/35 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/16 statements.
# Partially parsed test_unzip_empty_zipfile. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory. Retrieved 5/16 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 9/21 statements.
# Partially parsed test_unzip_password_protected_invalid_password. Retrieved 4/14 statements.


import _io as module_0

def test_case_0():
    var_0 = "Test unzip with a URL when file doesn't exist yet."
    var_1 = module_0.BytesIO()
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 1024
    var_8 = [var_5]
    var_9 = 'test_project/'
    var_10 = 'test_project/file.txt'
    var_11 = 'http://example.com/test.zip'
    var_12 = True

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with a local file path.'
    var_1 = 'myproject/'
    var_2 = 'myproject/file.txt'
    var_3 = '/path/to/local.zip'
    var_4 = False
    var_5 = module_0.unzip(var_3, var_4)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for empty zipfile.'
    var_1 = '/path/to/empty.zip'
    var_2 = False
    var_3 = module_0.unzip(var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises error when no top-level directory exists.'
    var_1 = 'file.txt'
    var_2 = '/path/to/bad.zip'
    var_3 = False
    var_4 = module_0.unzip(var_2, var_3)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for invalid zipfile.'
    var_1 = '/path/to/invalid.zip'
    var_2 = False
    var_3 = module_0.unzip(var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and valid password provided.'
    var_1 = 'project/'
    var_2 = 'project/file.txt'
    var_3 = 'Bad password'
    var_4 = None
    var_5 = '/path/to/protected.zip'
    var_6 = False
    var_7 = 'mypassword'
    var_8 = module_0.unzip(var_5, var_6, password=var_7)

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and invalid password provided.'
    var_1 = 'project/'
    var_2 = 'project/file.txt'
    var_3 = 'Bad password'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_unzip_bad_zip_file_exception. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'Test that BadZipFile exception at line 105 is caught and re-raised as InvalidZipRepository.'
    var_1 = 'fake.zip'
    var_2 = b'This is not a valid zip file'
    var_3 = 'clone'
    var_4 = False
    var_5 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_non_empty_zipfile. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 evaluates to False for non-empty zipfile.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 11/30 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 10/22 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 11/22 statements.
# Partially parsed test_unzip_password_protected_no_password_raises_error. Retrieved 12/25 statements.
# Partially parsed test_unzip_url_existing_file_no_input_deletes. Retrieved 12/31 statements.


import _io as module_0

def test_case_0():
    var_0 = 'Test unzip downloads and extracts a URL-based zipfile.'
    var_1 = module_0.BytesIO()
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip extracts a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = True
    var_9 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = b'not a zip file'
    var_3 = False
    var_4 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip extracts password-protected zip with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'utf-8'
    var_4 = module_0.encode(var_3)
    var_5 = 'test_project/'
    var_6 = ''
    var_7 = 'test_project/file.txt'
    var_8 = 'content'
    var_9 = False
    var_10 = True

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'Test unzip raises error for password-protected zip without password.'
    var_1 = 'protected.zip'
    var_2 = 'test_password'
    var_3 = 'utf-8'
    var_4 = module_0.encode(var_3)
    var_5 = 'test_project/'
    var_6 = ''
    var_7 = 'test_project/file.txt'
    var_8 = 'content'
    var_9 = False
    var_10 = True
    var_11 = None

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip deletes existing file when no_input=True.'
    var_1 = 'clone'
    var_2 = 'test.zip'
    var_3 = b'old content'
    var_4 = module_0.BytesIO()
    var_5 = 'test_project/'
    var_6 = ''
    var_7 = 'test_project/file.txt'
    var_8 = 'new content'
    var_9 = 0
    var_10 = 'http://example.com/test.zip'
    var_11 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_on_bad_zipfile. Retrieved 11/22 statements.


import locale as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that BadZipFile exception is caught and converted to InvalidZipRepository.'
    var_1 = 'clone'
    var_2 = var_0 / var_1
    var_3 = True
    var_4 = 'fake.zip'
    var_5 = var_2 / var_4
    var_6 = 'This is not a valid zip file'
    var_7 = module_0.str(var_5)
    var_8 = False
    var_9 = True
    var_10 = module_1.unzip(var_7, var_8, var_2, var_9)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_raises_invalid_zip_repository_when_zipfile_is_empty. Retrieved 6/16 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zipfile is empty.'
    var_1 = 'empty.zip'
    var_2 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_3 = module_0.patch(var_2)
    var_4 = False
    var_5 = module_1.unzip(var_0, var_4, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_iter_content_chunk_filter. Retrieved 11/33 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True for non-empty chunks.'
    var_1 = 'http://example.com/archive.zip'
    var_2 = 'archive.zip'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = b''
    var_6 = b'chunk3'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = 'project_name/'
    var_9 = 'project_name/file.txt'
    var_10 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 14/32 statements.


import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to False.'
    var_1 = b''
    var_2 = None
    var_3 = 'cookiecutter.zipfile.requests.get'
    var_4 = 'test_dir/'
    var_5 = 'cookiecutter.zipfile.ZipFile'
    var_6 = 'cookiecutter.zipfile.tempfile.mkdtemp'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'builtins.open'
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = '.'
    var_13 = module_1.unzip(var_10, var_11, var_12, var_11, var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 13/38 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True when chunk has content.'
    var_1 = 'clone'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = b'chunk3'
    var_5 = b''
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = 'test_project/'
    var_8 = False
    var_9 = []
    var_10 = 'http://example.com/test.zip'
    var_11 = True
    var_12 = len(var_9)
    assert var_12 == 3



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_unzip_empty_zipfile_raises_invalid_zip_repository. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 evaluates to True when zip is empty.'
    var_1 = 'empty.zip'
    var_2 = False



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False for empty chunks."
    var_1 = b''



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 40 (if chunk:) evaluates to True for non-empty chunks.'
    var_1 = b'test data'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_unzip_raises_on_empty_zipfile. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zipfile is empty.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False
    var_4 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 12/40 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/31 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/20 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/21 statements.
# Partially parsed test_unzip_with_password_protection. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = b'test'
    var_6 = [var_5]
    var_7 = 'project_name/'
    var_8 = 'project_name/file.txt'
    var_9 = False
    var_10 = 'http://example.com/test.zip'
    var_11 = True

def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'project_name/'
    var_2 = ''
    var_3 = 'project_name/file.txt'
    var_4 = 'content'
    var_5 = 'project_name/'
    var_6 = 'project_name/file.txt'
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com/test.zip'
    var_2 = False
    var_3 = True

def test_case_0():
    var_0 = 'file.txt'
    var_1 = False
    var_2 = 'http://example.com/test.zip'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'project_name/'
    var_1 = 'project_name/file.txt'
    var_2 = 'encrypted'
    var_3 = None
    var_4 = False
    var_5 = 'http://example.com/test.zip'
    var_6 = 'mypassword'



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 41 evaluates to False."
    var_1 = b''
    var_2 = bool(var_1)
    assert var_2 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 10/33 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 9/21 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 5/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 7/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_with_password_protected_zip. Retrieved 11/24 statements.
# Partially parsed test_unzip_creates_clone_to_dir_if_not_exists. Retrieved 11/26 statements.
# Partially parsed test_unzip_with_expanduser_path. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test unzip with a URL and new zipfile.'
    var_1 = 'zip_storage'
    var_2 = 'test.zip'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.requests.get'
    var_8 = 'http://example.com/test.zip'
    var_9 = True

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'zip_storage'
    var_2 = 'local_test.zip'
    var_3 = 'my_project/'
    var_4 = ''
    var_5 = 'my_project/README.md'
    var_6 = 'Project content'
    var_7 = False
    var_8 = True

def test_case_0():
    var_0 = 'Test unzip with empty zipfile raises InvalidZipRepository.'
    var_1 = 'zip_storage'
    var_2 = 'empty.zip'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'Test unzip with no top-level directory raises InvalidZipRepository.'
    var_1 = 'zip_storage'
    var_2 = 'no_toplevel.zip'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = 'Test unzip with invalid zipfile raises InvalidZipRepository.'
    var_1 = 'zip_storage'
    var_2 = 'invalid.zip'
    var_3 = 'This is not a zip file'
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'Test unzip with password-protected zipfile.'
    var_1 = 'zip_storage'
    var_2 = 'protected.zip'
    var_3 = 'secure_project/'
    var_4 = ''
    var_5 = 'secure_project/file.txt'
    var_6 = 'secret content'
    var_7 = b'test_password'
    var_8 = False
    var_9 = True
    var_10 = 'test_password'

def test_case_0():
    var_0 = "Test unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'zip_storage'
    var_2 = 'test.zip'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'content'
    var_7 = 'new_dir'
    var_8 = 'nested'
    var_9 = False
    var_10 = True

def test_case_0():
    var_0 = 'Test unzip expands user path correctly.'
    var_1 = 'zip_storage'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_unzip_iter_content_filters_out_keep_alive_chunks. Retrieved 12/30 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to False for keep-alive chunks.'
    var_1 = b''
    var_2 = None
    var_3 = b'test_data'
    var_4 = b'more_data'
    var_5 = 'requests.get'
    var_6 = 'project_dir/'
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = 'tempfile.mkdtemp'
    var_9 = 'builtins.open'
    var_10 = 'http://example.com/test.zip'
    var_11 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 9/23 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 9/18 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/15 statements.
# Partially parsed test_unzip_no_top_level_dir_raises_error. Retrieved 6/17 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_with_password_protected_zip_no_input. Retrieved 10/23 statements.
# Partially parsed test_unzip_with_correct_password. Retrieved 10/29 statements.
# Partially parsed test_unzip_expanduser_on_clone_to_dir. Retrieved 9/20 statements.


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

import genericpath as module_0

def test_case_0():
    var_0 = "Test that unzip creates clone_to_dir if it doesn't exist."
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'new_clone_dir'
    var_7 = False
    var_8 = module_0.exists()

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'clone'
    var_3 = False

def test_case_0():
    var_0 = 'Test that unzip raises error when no top-level directory exists.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = 'clone'
    var_5 = False

def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'not a zip file'
    var_3 = 'clone'
    var_4 = False

def test_case_0():
    var_0 = 'Test that unzip raises error for password protected zip with no_input.'
    var_1 = 'protected.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = b'password'
    var_7 = 'clone'
    var_8 = False
    var_9 = True

def test_case_0():
    var_0 = 'Test that unzip works with correct password.'
    var_1 = 'protected.zip'
    var_2 = 'content'
    var_3 = 'project_name'
    var_4 = 'file.txt'
    var_5 = [var_4]
    var_6 = 'password'
    var_7 = 5
    var_8 = 'clone'
    var_9 = False

def test_case_0():
    var_0 = 'Test that unzip expands user home directory.'
    var_1 = 'test.zip'
    var_2 = 'project_name/'
    var_3 = ''
    var_4 = 'project_name/file.txt'
    var_5 = 'content'
    var_6 = 'HOME'
    var_7 = False
    var_8 = '~/test_clone'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_unzip_with_local_file. Retrieved 11/23 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 7/16 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 9/18 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 8/15 statements.
# Partially parsed test_unzip_url_downloads_and_extracts. Retrieved 17/33 statements.
# Partially parsed test_unzip_url_existing_file_no_input_deletes_and_redownloads. Retrieved 18/34 statements.
# Partially parsed test_unzip_password_protected_with_correct_password. Retrieved 12/22 statements.
# Partially parsed test_unzip_password_protected_with_wrong_password_raises_error. Retrieved 14/25 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with a local zipfile.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = False
    var_10 = 'project_name'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for empty zip.'
    var_1 = 'empty.zip'
    var_2 = 'extract'
    var_3 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_4 = module_0.patch(var_3)
    var_5 = False
    var_6 = module_1.unzip(var_0, var_5, var_2)

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'no_toplevel.zip'
    var_2 = 'extract'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = False
    var_8 = module_1.unzip(var_3, var_7, var_2)

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip.'
    var_1 = 'invalid.zip'
    var_2 = 'extract'
    var_3 = 'not a zip file'
    var_4 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_5 = module_0.patch(var_4)
    var_6 = False
    var_7 = module_1.unzip(var_0, var_6, var_2)

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL downloads and extracts the file.'
    var_1 = 'remote.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.os.path.exists'
    var_10 = False
    var_11 = module_0.patch(var_9)
    var_12 = 'cookiecutter.zipfile.requests.get'
    var_13 = 'cookiecutter.zipfile.open'
    var_14 = 'http://example.com/test.zip'
    var_15 = True
    var_16 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with URL and existing file deletes it when no_input=True.'
    var_1 = 'remote.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_8 = module_0.patch(var_7)
    var_9 = 'cookiecutter.zipfile.os.path.exists'
    var_10 = True
    var_11 = module_0.patch(var_9)
    var_12 = 'cookiecutter.zipfile.prompt_and_delete'
    var_13 = module_0.patch(var_12)
    var_14 = 'cookiecutter.zipfile.requests.get'
    var_15 = 'cookiecutter.zipfile.open'
    var_16 = 'http://example.com/test.zip'
    var_17 = 'project_name'

import requests.api as module_0

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and correct password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'test_password'
    var_4 = 'project_name/'
    var_5 = ''
    var_6 = 'project_name/file.txt'
    var_7 = 'content'
    var_8 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_9 = module_0.patch(var_8)
    var_10 = False
    var_11 = 'project_name'

import requests.api as module_0
import cookiecutter.zipfile as module_1

def test_case_0():
    var_0 = 'Test unzip with password-protected zip and wrong password.'
    var_1 = 'protected.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_6 = module_0.patch(var_5)
    var_7 = 'cookiecutter.zipfile.ZipFile'
    var_8 = module_0.patch(var_7)
    var_9 = 'project_name/'
    var_10 = 'Bad password'
    var_11 = False
    var_12 = 'wrong'
    var_13 = module_1.unzip(var_3, var_11, var_2, password=var_12)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 9/35 statements.


def test_case_0():
    var_0 = "Test that the predicate 'if chunk:' at line 40 evaluates to False for empty chunks."
    var_1 = 'http://example.com/test.zip'
    var_2 = b''
    var_3 = b'data'
    var_4 = 'project_dir/'
    var_5 = []
    var_6 = True
    var_7 = b''
    var_8 = 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_unzip_writes_chunks_to_file. Retrieved 17/37 statements.


import genericpath as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 (if chunk:) evaluates to True for valid chunks.'
    var_1 = 'test.zip'
    var_2 = 'extract'
    var_3 = 'project_name/'
    var_4 = ''
    var_5 = 'project_name/file.txt'
    var_6 = 'content'
    var_7 = 'clone'
    var_8 = b'test_chunk_1'
    var_9 = b'test_chunk_2'
    var_10 = b''
    var_11 = b'test_chunk_3'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = 'http://example.com/test.zip'
    var_14 = True
    var_15 = None
    var_16 = module_0.exists()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_unzip_with_url_new_file. Retrieved 18/38 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 8/18 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/12 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 5/14 statements.
# Partially parsed test_unzip_with_password_provided. Retrieved 9/19 statements.
# Partially parsed test_unzip_invalid_zip_raises_error. Retrieved 4/11 statements.
# Partially parsed test_unzip_with_url_existing_file_no_input. Retrieved 24/48 statements.


import _io as module_0

def test_case_0():
    var_0 = "Test unzip with URL when zip file doesn't exist."
    var_1 = module_0.BytesIO()
    var_2 = 'test_project/'
    var_3 = ''
    var_4 = 'test_project/file.txt'
    var_5 = 'content'
    var_6 = 0
    var_7 = 'clone'
    var_8 = 'MockResponse'
    var_9 = ()
    var_10 = 'iter_content'
    var_11 = 'cookiecutter.zipfile.requests.get'
    var_12 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_13 = None
    var_14 = lambda x: var_13
    var_15 = 'http://example.com/test.zip'
    var_16 = True
    var_17 = 'test_project'

def test_case_0():
    var_0 = 'Test unzip with local file path.'
    var_1 = 'test.zip'
    var_2 = 'my_project/'
    var_3 = ''
    var_4 = 'my_project/README.md'
    var_5 = 'test content'
    var_6 = False
    var_7 = 'my_project'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when zip is empty.'
    var_1 = 'empty.zip'
    var_2 = False

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository when no top-level directory.'
    var_1 = 'notoplevel.zip'
    var_2 = 'file.txt'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test unzip with password provided.'
    var_1 = 'protected.zip'
    var_2 = 'secure_project/'
    var_3 = ''
    var_4 = 'secure_project/file.txt'
    var_5 = 'secret'
    var_6 = b'mypassword'
    var_7 = False
    var_8 = 'mypassword'

def test_case_0():
    var_0 = 'Test unzip raises InvalidZipRepository for invalid zip file.'
    var_1 = 'invalid.zip'
    var_2 = 'This is not a zip file'
    var_3 = False

import _io as module_0

def test_case_0():
    var_0 = 'Test unzip with URL when file exists and no_input=True.'
    var_1 = 'clone'
    var_2 = 'existing.zip'
    var_3 = 'project/'
    var_4 = ''
    var_5 = 'project/file.txt'
    var_6 = 'old content'
    var_7 = module_0.BytesIO()
    var_8 = 'project/'
    var_9 = ''
    var_10 = 'project/file.txt'
    var_11 = 'new content'
    var_12 = 0
    var_13 = 'MockResponse'
    var_14 = ()
    var_15 = 'iter_content'
    var_16 = 'cookiecutter.zipfile.requests.get'
    var_17 = 'cookiecutter.zipfile.make_sure_path_exists'
    var_18 = None
    var_19 = lambda x: var_18
    var_20 = 'cookiecutter.zipfile.os.path.exists'
    var_21 = True
    var_22 = lambda x: var_21
    var_23 = 'http://example.com/existing.zip'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_unzip_downloads_zipfile_in_chunks. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 40 evaluates to True when chunk is not empty.'
    var_1 = b'test chunk data'
    var_2 = b''
    var_3 = 'test_dir/'
    var_4 = ''
    var_5 = 'test_dir/file.txt'
    var_6 = 'content'
    var_7 = []
    var_8 = 'http://example.com/test.zip'
    var_9 = True
    var_10 = len(var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_unzip_download_predicate_false_when_not_url. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False when is_url is False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = 'clone'
    var_7 = False
    var_8 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 5/23 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 4/17 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 2/14 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 3/15 statements.
# Partially parsed test_unzip_with_password. Retrieved 6/19 statements.
# Partially parsed test_unzip_bad_zip_file_raises_error. Retrieved 2/13 statements.
# Partially parsed test_unzip_creates_clone_to_dir. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'https://example.com/repo.zip'
    var_1 = b'test'
    var_2 = 'project_name/'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = 'project_name/'
    var_2 = False
    var_3 = 'project_name'

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = 'file.txt'
    var_2 = False

import builtins as module_0

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = 'test_password'
    var_2 = 'project_name/'
    var_3 = module_0.RuntimeError()
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = '/local/path/repo.zip'
    var_1 = False

def test_case_0():
    var_0 = 'new_dir'
    var_1 = '/local/path/repo.zip'
    var_2 = 'project_name/'
    var_3 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_unzip_predicate_line_55_evaluates_to_false. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 55 (len(zip_file.namelist()) == 0) evaluates to False.'
    var_1 = 'test.zip'
    var_2 = 'project_dir/'
    var_3 = ''
    var_4 = 'project_dir/file.txt'
    var_5 = 'content'
    var_6 = False
    var_7 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_unzip_with_url_downloads_and_extracts. Retrieved 8/32 statements.
# Partially parsed test_unzip_with_local_file_extracts. Retrieved 5/22 statements.
# Partially parsed test_unzip_with_password_protected_zip_prompts_user. Retrieved 10/33 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 4/23 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/21 statements.


def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = [var_0, var_1]
    var_3 = 'myproject/'
    var_4 = 'myproject/file.txt'
    var_5 = 'http://example.com/myproject.zip'
    var_6 = True
    var_7 = 'myproject'

def test_case_0():
    var_0 = 'myproject/'
    var_1 = 'myproject/file.txt'
    var_2 = '/path/to/local.zip'
    var_3 = False
    var_4 = 'myproject'

def test_case_0():
    var_0 = b'chunk1'
    var_1 = [var_0]
    var_2 = 'myproject/'
    var_3 = 'myproject/file.txt'
    var_4 = 'Bad password'
    var_5 = None
    var_6 = 'http://example.com/myproject.zip'
    var_7 = True
    var_8 = False
    var_9 = 'myproject'

def test_case_0():
    var_0 = b'chunk1'
    var_1 = [var_0]
    var_2 = 'http://example.com/empty.zip'
    var_3 = True

def test_case_0():
    var_0 = b'chunk1'
    var_1 = [var_0]
    var_2 = 'file.txt'
    var_3 = 'another_file.txt'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_unzip_with_url_creates_directory. Retrieved 8/25 statements.
# Partially parsed test_unzip_with_local_file. Retrieved 6/15 statements.
# Partially parsed test_unzip_empty_zip_raises_error. Retrieved 3/13 statements.
# Partially parsed test_unzip_no_top_level_directory_raises_error. Retrieved 4/14 statements.
# Partially parsed test_unzip_password_protected_with_valid_password. Retrieved 8/18 statements.
# Partially parsed test_unzip_password_protected_invalid_password_raises_error. Retrieved 7/18 statements.
# Partially parsed test_unzip_url_with_existing_file_and_delete. Retrieved 2/5 statements.


import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project_dir/'
    var_1 = 'project_dir/file.txt'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = 'http://example.com/repo.zip'
    var_5 = True
    var_6 = '.'
    var_7 = module_0.unzip(var_4, var_5, var_6, var_5)
    assert var_7 == '/tmp/test/project_dir'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'myproject/'
    var_1 = 'myproject/file.txt'
    var_2 = '/local/repo.zip'
    var_3 = False
    var_4 = '.'
    var_5 = module_0.unzip(var_2, var_3, var_4)
    assert var_5 == '/tmp/test/myproject'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = '/local/repo.zip'
    var_2 = False
    var_3 = module_0.unzip(var_1, var_2)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project/'
    var_1 = 'project/file.txt'
    var_2 = 'Bad password'
    var_3 = None
    var_4 = '/local/repo.zip'
    var_5 = False
    var_6 = 'mypassword'
    var_7 = module_0.unzip(var_4, var_5, password=var_6)
    assert var_7 == '/tmp/test/project'

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = 'project/'
    var_1 = 'project/file.txt'
    var_2 = 'Bad password'
    var_3 = '/local/repo.zip'
    var_4 = False
    var_5 = 'wrongpassword'
    var_6 = module_0.unzip(var_3, var_4, password=var_5)

import cookiecutter.zipfile as module_0

def test_case_0():
    var_0 = '/local/repo.zip'
    var_1 = False
    var_2 = module_0.unzip(var_0, var_1)

def test_case_0():
    var_0 = 'project/'
    var_1 = 'project/file.txt'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unzip_raises_on_empty_zipfile. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'Test that unzip raises InvalidZipRepository when zip file is empty.'
    var_1 = 'empty.zip'
    var_2 = False



