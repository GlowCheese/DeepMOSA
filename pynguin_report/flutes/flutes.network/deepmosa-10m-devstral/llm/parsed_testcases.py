####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_download_from_direct_url. Retrieved 5/6 statements.
# Partially parsed test_download_from_google_drive. Retrieved 5/6 statements.
# Partially parsed test_download_with_default_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_github_url. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract. Retrieved 7/9 statements.
# Partially parsed test_download_with_progress. Retrieved 6/7 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 7/8 statements.
# Partially parsed test_download_with_existing_file. Retrieved 6/11 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'file.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = [var_2]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = '/tmp'
    var_2 = '123456789'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = [var_2]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'file.txt'
    var_7 = [var_6]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.txt?raw=true'
    var_1 = '/tmp'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'file.txt'
    var_7 = [var_6]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp'
    var_2 = 'archive.tar.gz'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, var_3, **var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'extracted_file'
    var_9 = [var_8]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'file.txt'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = [var_2]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'file.txt'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, bar_fn=var_4, **var_5)
    var_7 = module_1.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = [var_2]

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'file.txt'
    var_3 = [var_2]
    var_4 = 'existing content'
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, **var_5)
    var_7 = module_1.exists(var_6)
    assert var_7 == 'existing content'
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__download_from_google_drive. Retrieved 6/7 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = './test_dir'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = [var_1]



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = var_0 is var_0
    var_2 = var_0 if var_1 else var_0
    assert var_2 is None



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = b''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_progress_close_is_called_when_bar_fn_is_not_none. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345/view'
    var_1 = 'test.txt'
    var_2 = '/tmp'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_extract_tarfile. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test.tar.gz'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__download_without_progress_bar. Retrieved 5/6 statements.
# Partially parsed test__download_with_progress_bar. Retrieved 3/10 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = [var_1]
    var_5 = module_1.exists(var_3)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------




import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = module_2.basename(var_4)
    var_8 = bool(var_7 == var_1)
    assert var_8 is True
    var_9 = module_2.dirname(var_4)
    var_10 = bool(var_9 == var_2)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_progress_close_is_called_when_bar_fn_is_not_none. Retrieved 3/24 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123'
    var_1 = 'test.txt'
    var_2 = '/tmp'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_28. Retrieved 1/2 statements.


def test_case_0():
    var_0 = b'data'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__download_from_google_drive. Retrieved 5/6 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1abc123/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download_from_google_drive(var_0, var_1, var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = [var_1]



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_progress_close_is_called. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123'
    var_1 = 'test.txt'
    var_2 = '/tmp'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_download_from_google_drive. Retrieved 3/17 statements.


def test_case_0():
    var_0 = b'test data'
    assert var_0 == b'test data'
    var_1 = 'https://drive.google.com/d/test_file_id/view'
    var_2 = 'test_file.txt'
    var_3 = bool(var_0)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 6/7 statements.
# Partially parsed test_download_from_google_drive_with_progress_bar. Retrieved 3/8 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = [var_1]

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_token_present_in_response_cookies. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'cookies'
    var_3 = 'download_warning_token'
    var_4 = 'test_token'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__download_without_progress_bar. Retrieved 4/7 statements.
# Partially parsed test__download_with_progress_bar. Retrieved 3/9 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = [var_1]
    var_5 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]
    var_4 = [var_1]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_progress_close_called. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345'
    var_1 = 'test.txt'
    var_2 = '/tmp'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_download_with_none_save_dir. Retrieved 3/5 statements.
# Partially parsed test_download_with_custom_save_dir. Retrieved 2/5 statements.
# Partially parsed test_download_with_custom_filename. Retrieved 2/5 statements.
# Partially parsed test_download_with_google_drive_url. Retrieved 2/5 statements.
# Partially parsed test_download_with_github_url. Retrieved 2/5 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 3/6 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 3/6 statements.
# Partially parsed test_download_with_progress. Retrieved 3/6 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 2/7 statements.
# Partially parsed test_download_with_existing_file. Retrieved 3/8 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)
    var_3 = 'test.txt'

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'custom.txt'

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = '123456789'

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/test.txt?raw=true'
    var_1 = 'test.txt'

def test_case_0():
    var_0 = 'https://example.com/test.tar.gz'
    var_1 = True
    var_2 = 'test.tar.gz'

def test_case_0():
    var_0 = 'https://example.com/test.zip'
    var_1 = True
    var_2 = 'test.zip'

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = True
    var_2 = 'test.txt'

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'
    var_2 = 'https://example.com/test.txt'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_progress_not_none.




# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__download_without_progress_bar. Retrieved 5/6 statements.
# Partially parsed test__download_with_progress_bar. Retrieved 3/8 statements.
# Partially parsed test__download_with_progress_bar_and_total_size. Retrieved 3/11 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = [var_1]
    var_5 = module_1.exists(var_3)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__download_from_google_drive_with_progress_bar. Retrieved 5/8 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file.txt'
    var_5 = '/tmp/test_file.txt'
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import genericpath as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = '/tmp/test_file.txt'
    var_4 = module_0.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://invalid.url'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)



# Parsed testcases at query #28
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'test.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    assert var_4 == '/tmp/test.txt'
    var_5 = 'https://drive.google.com/file/d/12345/view'
    var_6 = {}
    var_7 = module_0.download(var_5, var_1, var_2, **var_6)
    assert var_7 == '/tmp/test.txt'
    var_8 = 'https://github.com/user/repo/raw/main/file.txt'
    var_9 = {}
    var_10 = module_0.download(var_8, var_1, **var_9)
    assert var_10 == '/tmp/file.txt'
    var_11 = True
    var_12 = {}
    var_13 = module_0.download(var_0, var_1, var_2, var_11, **var_12)
    assert var_13 == '/tmp/test.txt'
    var_14 = {}
    var_15 = module_0.download(var_0, var_1, var_2, progress=var_11, **var_14)
    assert var_15 == '/tmp/test.txt'
    var_16 = None
    var_17 = lambda : var_16
    var_18 = {}
    var_19 = module_0.download(var_0, var_1, var_2, bar_fn=var_17, **var_18)
    assert var_19 == '/tmp/test.txt'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__progress_hook_returns_true. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 1
    var_2 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_download_from_google_drive. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharing'
    var_1 = 'test_file.txt'
    var_2 = b'test content'
    assert var_2 == b'test content'



# Parsed testcases at query #31
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'requests'
    var_1 = globals()
    var_2 = var_0 in var_1
    var_3 = locals()
    var_4 = var_0 in var_3
    var_5 = bool(var_2 or var_4)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_progress_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #34
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #35
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1234567890abcdef/view?usp=sharing'
    var_1 = 'test_file'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test_file'



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = 'mock_progress'
    var_2 = lambda : var_1
    var_3 = var_2()
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_token_predicate_evaluates_to_true. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view?usp=sharing'
    var_1 = 'test_file'
    var_2 = '/tmp'
    var_3 = None
    var_4 = 'download_warning_123'
    var_5 = 'confirm_token'
    var_6 = b'chunk1'
    var_7 = b'chunk2'
    var_8 = [var_6, var_7]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__download_basic. Retrieved 4/7 statements.
# Partially parsed test__download_with_progress_bar. Retrieved 3/9 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = [var_1]
    var_5 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]
    var_4 = [var_1]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test__download_from_google_drive. Retrieved 4/19 statements.


def test_case_0():
    var_0 = b'test data'
    var_1 = [var_0]
    var_2 = 'https://drive.google.com/d/test_id/view'
    assert var_2 == b'test data'
    var_3 = 'test_file.txt'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_progress_is_not_none_when_bar_fn_is_provided. Retrieved 8/30 statements.


def test_case_0():
    var_0 = 'http://example.com/file'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = 0
    var_5 = 1
    var_6 = 1024
    var_7 = 2048
    var_8 = bool(var_3 is not None)
    assert var_8 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__download_without_progress_bar. Retrieved 4/7 statements.
# Partially parsed test__download_with_progress_bar. Retrieved 3/9 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = module_0._download(var_0, var_1, var_2)
    var_4 = [var_1]
    var_5 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]
    var_4 = [var_1]



# Parsed testcases at query #2
#--------------------------




import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '123456789'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789/edit?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '123456789'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/123456789'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '123456789'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_download_from_google_drive. Retrieved 6/7 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1abc123/view'
    var_1 = 'test_file'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    var_5 = module_1.exists(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = [var_1]



