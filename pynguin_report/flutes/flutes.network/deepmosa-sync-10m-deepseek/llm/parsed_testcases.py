####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_download_with_default_filename. Retrieved 2/14 statements.
# Partially parsed test_download_with_custom_filename. Retrieved 3/15 statements.
# Partially parsed test_download_google_drive_url. Retrieved 2/14 statements.
# Partially parsed test_download_existing_file_skips. Retrieved 3/13 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 3/15 statements.
# Partially parsed test_download_with_extraction_tar. Retrieved 5/25 statements.
# Partially parsed test_download_with_extraction_zip. Retrieved 5/25 statements.
# Partially parsed test_download_github_raw_url. Retrieved 2/14 statements.
# Partially parsed test_download_no_save_dir_uses_temp. Retrieved 4/15 statements.
# Partially parsed test_download_google_drive_with_token. Retrieved 5/19 statements.


def test_case_0():
    var_0 = b'data'
    var_1 = 'http://example.com/file.txt'

def test_case_0():
    var_0 = b'data'
    var_1 = 'http://example.com/file.txt'
    var_2 = 'custom.bin'

def test_case_0():
    var_0 = b'data'
    var_1 = 'https://drive.google.com/file/d/abc123/view'

def test_case_0():
    var_0 = 'existing.txt'
    var_1 = 'content'
    var_2 = 'http://example.com/existing.txt'

def test_case_0():
    var_0 = b'data'
    var_1 = 'http://example.com/file.txt'
    var_2 = True

def test_case_0():
    var_0 = b'data'
    var_1 = 'archive.tar.gz'
    var_2 = b'fake tar content'
    var_3 = 'http://example.com/archive.tar.gz'
    var_4 = True

def test_case_0():
    var_0 = b'data'
    var_1 = 'archive.zip'
    var_2 = b'fake zip content'
    var_3 = 'http://example.com/archive.zip'
    var_4 = True

def test_case_0():
    var_0 = b'data'
    var_1 = 'https://github.com/user/repo/raw/main/script.py?raw=true'

import flutes.network as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = 'http://example.com/tempfile.bin'
    var_2 = None
    var_3 = {}
    var_4 = module_0.download(var_1, var_2, **var_3)

def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc'
    assert var_1 == 'def456'
    var_2 = (var_0, var_1)
    var_3 = b'data'
    var_4 = 'https://drive.google.com/file/d/def456/view'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__download_from_google_drive_success. Retrieved 30/49 statements.
# Partially parsed test__download_from_google_drive_with_token. Retrieved 37/63 statements.
# Partially parsed test__download_from_google_drive_with_progress. Retrieved 42/68 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = 'OpenMock'
    var_18 = ()
    var_19 = '__enter__'
    var_20 = '__exit__'
    var_21 = 'write'
    var_22 = lambda self: self
    var_23 = None
    var_24 = lambda self, exc_type, exc_val, exc_tb: var_23
    var_25 = lambda self, chunk: var_23
    var_26 = {var_19: var_22, var_20: var_24, var_21: var_25}
    var_27 = [var_17, var_18, var_26]
    var_28 = 'https://drive.google.com/file/d/abc123/view'
    var_29 = 'file.txt'
    var_30 = '/tmp'
    var_31 = module_0._download_from_google_drive(var_28, var_29, var_30, var_23)
    assert var_31 == '/tmp/file.txt'

import flutes.network as module_0

def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'xyz'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'iter_content'
    var_6 = 'cookies'
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_5: var_9, var_6: var_2}
    var_11 = [var_3, var_4, var_10]
    var_12 = ()
    var_13 = [var_7]
    var_14 = lambda chunk_size: var_13
    var_15 = {}
    var_16 = {var_5: var_14, var_6: var_15}
    var_17 = [var_3, var_12, var_16]
    var_18 = 0
    assert var_18 == 2
    var_19 = 'MockSession'
    var_20 = ()
    var_21 = 'get'
    var_22 = 'MockRequests'
    var_23 = ()
    var_24 = 'Session'
    var_25 = 'OpenMock'
    var_26 = ()
    var_27 = '__enter__'
    var_28 = '__exit__'
    var_29 = 'write'
    var_30 = lambda self: self
    var_31 = None
    var_32 = lambda self, exc_type, exc_val, exc_tb: var_31
    var_33 = lambda self, chunk: var_31
    var_34 = {var_27: var_30, var_28: var_32, var_29: var_33}
    var_35 = [var_25, var_26, var_34]
    var_36 = 'https://drive.google.com/file/d/def456/view'
    var_37 = 'file.txt'
    var_38 = '/tmp'
    var_39 = module_0._download_from_google_drive(var_36, var_37, var_38, var_31)
    assert var_39 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = 'OpenMock'
    var_18 = ()
    var_19 = '__enter__'
    var_20 = '__exit__'
    var_21 = 'write'
    var_22 = lambda self: self
    var_23 = None
    var_24 = lambda self, exc_type, exc_val, exc_tb: var_23
    var_25 = lambda self, chunk: var_23
    var_26 = {var_19: var_22, var_20: var_24, var_21: var_25}
    var_27 = [var_17, var_18, var_26]
    var_28 = []
    var_29 = False
    assert var_29 is True
    var_30 = 'MockProgress'
    var_31 = ()
    var_32 = 'update'
    var_33 = 'close'
    var_34 = lambda size: var_28.append(size)
    var_35 = globals()
    var_36 = 'progress_close_called'
    var_37 = True
    var_38 = 'https://drive.google.com/file/d/ghi789/view'
    var_39 = 'file.txt'
    var_40 = '/tmp'
    var_41 = len(var_4)
    var_42 = len(var_5)
    var_43 = [var_41, var_42]
    var_44 = bool(var_28 == var_43)
    assert var_44 is True

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/def456/preview/extra'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'def456'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/ghi789'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'ghi789'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_false. Retrieved 6/19 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345/view'
    var_1 = 'test.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = b''
    var_5 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_5 == '/tmp/test.txt'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_download_from_google_drive_success_with_bar. Retrieved 3/11 statements.
# Partially parsed test_download_from_google_drive_success_without_bar. Retrieved 5/10 statements.
# Partially parsed test_download_from_google_drive_no_token. Retrieved 6/11 statements.
# Partially parsed test_download_from_google_drive_extract_id. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = 'file.txt'
    var_2 = '/tmp'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/d/xyz456'
    var_1 = 'data.bin'
    var_2 = '/home/user'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/home/user/data.bin'

import flutes.network as module_0

def test_case_0():
    var_0 = False
    var_1 = 'https://drive.google.com/d/def789'
    var_2 = 'out.txt'
    var_3 = '/var'
    var_4 = None
    var_5 = module_0._download_from_google_drive(var_1, var_2, var_3, var_4)
    assert var_5 == '/var/out.txt'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/complex_id_123_abc/details'
    var_1 = 'test.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download_from_google_drive(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/test.txt'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_download_from_google_drive_with_valid_url_and_bar_fn. Retrieved 19/40 statements.
# Partially parsed test_download_from_google_drive_without_bar_fn. Retrieved 15/29 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 23/44 statements.
# Partially parsed test_download_from_google_drive_with_empty_chunks. Retrieved 20/41 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'update'
    var_2 = 'close'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = lambda self: var_3
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = 'iter_content'
    var_8 = b'data'
    var_9 = [var_8]
    var_10 = lambda self, chunk_size: var_9
    var_11 = {var_7: var_10}
    var_12 = 'get'
    var_13 = 'requests'
    var_14 = __import__(var_13)
    var_15 = 'Session'
    var_16 = 'https://drive.google.com/file/d/12345/view'
    var_17 = 'file.txt'
    var_18 = '/tmp'

import flutes.network as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = 'iter_content'
    var_2 = b'data'
    var_3 = [var_2]
    var_4 = lambda self, chunk_size: var_3
    var_5 = {var_1: var_4}
    var_6 = 'get'
    var_7 = 'requests'
    var_8 = __import__(var_7)
    var_9 = 'Session'
    var_10 = 'https://drive.google.com/file/d/12345/view'
    var_11 = 'file.txt'
    var_12 = '/tmp'
    var_13 = None
    var_14 = module_0._download_from_google_drive(var_10, var_11, var_12, var_13)
    assert var_14 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'obj'
    var_1 = 'update'
    var_2 = 'close'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = lambda self: var_3
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = 'cookies'
    var_8 = 'iter_content'
    var_9 = 'download_warning_token'
    var_10 = 'abc'
    var_11 = {var_9: var_10}
    var_12 = b'data'
    var_13 = [var_12]
    var_14 = lambda self, chunk_size: var_13
    var_15 = {var_7: var_11, var_8: var_14}
    var_16 = 'get'
    var_17 = 'requests'
    var_18 = __import__(var_17)
    var_19 = 'Session'
    var_20 = 'https://drive.google.com/file/d/12345/view'
    var_21 = 'file.txt'
    var_22 = '/tmp'

def test_case_0():
    var_0 = 'obj'
    var_1 = 'update'
    var_2 = 'close'
    var_3 = None
    var_4 = lambda self, x: var_3
    var_5 = lambda self: var_3
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = 'iter_content'
    var_8 = b''
    var_9 = b'data'
    var_10 = [var_8, var_9, var_8]
    var_11 = lambda self, chunk_size: var_10
    var_12 = {var_7: var_11}
    var_13 = 'get'
    var_14 = 'requests'
    var_15 = __import__(var_14)
    var_16 = 'Session'
    var_17 = 'https://drive.google.com/file/d/12345/view'
    var_18 = 'file.txt'
    var_19 = '/tmp'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_download_with_direct_url. Retrieved 5/6 statements.
# Partially parsed test_download_with_google_drive_url. Retrieved 6/7 statements.
# Partially parsed test_download_with_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_without_save_dir. Retrieved 4/6 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_existing_file. Retrieved 7/11 statements.
# Partially parsed test_download_with_extraction. Retrieved 6/7 statements.
# Partially parsed test_download_github_raw_url. Retrieved 6/7 statements.
# Partially parsed test_download_with_bar_fn. Retrieved 3/18 statements.
# Partially parsed test_download_google_drive_with_bar_fn. Retrieved 4/19 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'downloaded.txt'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test'
    var_2 = 'abc123'
    var_3 = None
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_3, progress=var_4, **var_5)
    var_7 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'custom.txt'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = None
    var_2 = False
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_1, progress=var_2, **var_3)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'file.txt'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = False
    var_7 = {}
    var_8 = module_0.download(var_0, var_1, var_2, progress=var_6, **var_7)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = 'archive.tar.gz'
    var_3 = True
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, var_3, var_4, **var_5)
    var_7 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.txt?raw=true'
    var_1 = '/tmp/test'
    var_2 = 'file.txt'
    var_3 = None
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_3, progress=var_4, **var_5)
    var_7 = [var_2]

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'file.txt'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/xyz789/view'
    var_1 = '/tmp/test'
    var_2 = 'xyz789'
    var_3 = None
    var_4 = [var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_download_with_direct_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_direct_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_github_raw_url. Retrieved 4/5 statements.
# Partially parsed test_download_with_google_drive_url. Retrieved 4/5 statements.
# Partially parsed test_download_with_google_drive_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_none_save_dir. Retrieved 3/5 statements.
# Partially parsed test_download_with_existing_file. Retrieved 6/10 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 3/18 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 5/6 statements.
# Partially parsed test_download_with_unknown_compression_type. Retrieved 5/6 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'data.txt'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test'
    var_2 = 'custom.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.txt'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.py?raw=true'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.py'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'abc123'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test'
    var_2 = 'file.zip'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'file.zip'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = None
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = {}
    var_7 = module_0.download(var_0, var_1, var_2, **var_6)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'data.txt'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test'
    var_2 = 'data.txt'
    var_3 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.tar.gz'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.zip'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/unknown.rar'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'unknown.rar'
    var_6 = [var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_download_without_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 3/19 statements.
# Partially parsed test_download_with_progress_bar_and_total_size. Retrieved 3/26 statements.
# Partially parsed test_download_with_progress_bar_and_unknown_total. Retrieved 3/26 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download(var_0, var_1, var_2, var_3)
    var_5 = [var_1]

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

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = [var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 30/49 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 37/63 statements.
# Partially parsed test_download_from_google_drive_with_progress. Retrieved 43/66 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda self, chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = 'OpenMock'
    var_18 = ()
    var_19 = '__enter__'
    var_20 = '__exit__'
    var_21 = 'write'
    var_22 = lambda self: self
    var_23 = None
    var_24 = lambda self, *args: var_23
    var_25 = lambda self, chunk: var_23
    var_26 = {var_19: var_22, var_20: var_24, var_21: var_25}
    var_27 = [var_17, var_18, var_26]
    var_28 = 'https://drive.google.com/file/d/12345/view'
    var_29 = 'file.txt'
    var_30 = '/tmp'
    var_31 = module_0._download_from_google_drive(var_28, var_29, var_30, var_23)
    assert var_31 == '/tmp/file.txt'

import flutes.network as module_0

def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'iter_content'
    var_6 = 'cookies'
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda self, chunk_size: var_8
    var_10 = {var_5: var_9, var_6: var_2}
    var_11 = [var_3, var_4, var_10]
    var_12 = ()
    var_13 = [var_7]
    var_14 = lambda self, chunk_size: var_13
    var_15 = {}
    var_16 = {var_5: var_14, var_6: var_15}
    var_17 = [var_3, var_12, var_16]
    var_18 = 0
    assert var_18 == 2
    var_19 = 'MockSession'
    var_20 = ()
    var_21 = 'get'
    var_22 = 'MockRequests'
    var_23 = ()
    var_24 = 'Session'
    var_25 = 'OpenMock'
    var_26 = ()
    var_27 = '__enter__'
    var_28 = '__exit__'
    var_29 = 'write'
    var_30 = lambda self: self
    var_31 = None
    var_32 = lambda self, *args: var_31
    var_33 = lambda self, chunk: var_31
    var_34 = {var_27: var_30, var_28: var_32, var_29: var_33}
    var_35 = [var_25, var_26, var_34]
    var_36 = 'https://drive.google.com/file/d/67890/view'
    var_37 = 'file.txt'
    var_38 = '/tmp'
    var_39 = module_0._download_from_google_drive(var_36, var_37, var_38, var_31)
    assert var_39 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda self, chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = 'OpenMock'
    var_18 = ()
    var_19 = '__enter__'
    var_20 = '__exit__'
    var_21 = 'write'
    var_22 = lambda self: self
    var_23 = None
    var_24 = lambda self, *args: var_23
    var_25 = lambda self, chunk: var_23
    var_26 = {var_19: var_22, var_20: var_24, var_21: var_25}
    var_27 = [var_17, var_18, var_26]
    var_28 = []
    var_29 = False
    var_30 = 'MockProgress'
    var_31 = ()
    var_32 = 'update'
    var_33 = 'close'
    var_34 = lambda self, size: var_28.append(size)
    var_35 = 'close_called'
    var_36 = True
    var_37 = lambda self: setattr(self, var_35, var_36)
    var_38 = {var_32: var_34, var_33: var_37}
    var_39 = [var_30, var_31, var_38]
    var_40 = 'https://drive.google.com/file/d/12345/view'
    var_41 = 'file.txt'
    var_42 = '/tmp'
    var_43 = len(var_28)
    assert var_43 == 2
    var_44 = len(var_4)
    var_45 = var_28[0]
    var_46 = bool(var_28[0] == var_44)
    assert var_46 is True
    var_47 = len(var_5)
    var_48 = var_28[1]
    var_49 = bool(var_28[1] == var_47)
    assert var_49 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 31/49 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 32/50 statements.
# Partially parsed test_download_from_google_drive_with_progress_bar. Retrieved 36/58 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = None
    var_18 = 'MockFile'
    var_19 = ()
    var_20 = 'write'
    var_21 = '__enter__'
    var_22 = '__exit__'
    var_23 = lambda data: var_17
    var_24 = lambda self: self
    var_25 = lambda self, exc_type, exc_val, exc_tb: var_17
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = [var_18, var_19, var_26]
    var_28 = 'test_path/test_file'
    var_29 = 'https://drive.google.com/file/d/12345/view'
    var_30 = 'test_file'
    var_31 = 'test_path'
    var_32 = module_0._download_from_google_drive(var_29, var_30, var_31, var_17)
    assert var_32 == 'test_path/test_file'

import flutes.network as module_0

def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc123'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'iter_content'
    var_6 = 'cookies'
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_5: var_9, var_6: var_2}
    var_11 = [var_3, var_4, var_10]
    var_12 = 'MockSession'
    var_13 = ()
    var_14 = 'get'
    var_15 = 'MockRequests'
    var_16 = ()
    var_17 = 'Session'
    var_18 = None
    var_19 = 'MockFile'
    var_20 = ()
    var_21 = 'write'
    var_22 = '__enter__'
    var_23 = '__exit__'
    var_24 = lambda data: var_18
    var_25 = lambda self: self
    var_26 = lambda self, exc_type, exc_val, exc_tb: var_18
    var_27 = {var_21: var_24, var_22: var_25, var_23: var_26}
    var_28 = [var_19, var_20, var_27]
    var_29 = 'path/file'
    var_30 = 'https://drive.google.com/d/67890'
    var_31 = 'file'
    var_32 = 'path'
    var_33 = module_0._download_from_google_drive(var_30, var_31, var_32, var_18)
    assert var_33 == 'path/file'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'data'
    var_5 = [var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {}
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = 'MockSession'
    var_11 = ()
    var_12 = 'get'
    var_13 = 'MockRequests'
    var_14 = ()
    var_15 = 'Session'
    var_16 = None
    var_17 = 'MockFile'
    var_18 = ()
    var_19 = 'write'
    var_20 = '__enter__'
    var_21 = '__exit__'
    var_22 = lambda data: var_16
    var_23 = lambda self: self
    var_24 = lambda self, exc_type, exc_val, exc_tb: var_16
    var_25 = {var_19: var_22, var_20: var_23, var_21: var_24}
    var_26 = [var_17, var_18, var_25]
    var_27 = 'dest/file'
    var_28 = 'MockBar'
    var_29 = ()
    var_30 = 'update'
    var_31 = 'close'
    var_32 = lambda size: var_16
    var_33 = lambda : var_16
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = [var_28, var_29, var_34]
    var_36 = 'https://drive.google.com/d/abc123'
    var_37 = 'file'
    var_38 = 'dest'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e/view?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == '1a2b3c4d5e'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/d/xyz789'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'xyz789'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 30/49 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 37/63 statements.
# Partially parsed test_download_from_google_drive_with_progress. Retrieved 37/60 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda self, chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = None
    var_18 = 'MockOpen'
    var_19 = ()
    var_20 = '__enter__'
    var_21 = '__exit__'
    var_22 = 'write'
    var_23 = lambda self: self
    var_24 = lambda self, exc_type, exc_val, exc_tb: var_17
    var_25 = lambda self, data: var_17
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = [var_18, var_19, var_26]
    var_28 = 'https://drive.google.com/file/d/abc123/view'
    var_29 = 'file.txt'
    var_30 = '/tmp'
    var_31 = module_0._download_from_google_drive(var_28, var_29, var_30, var_17)
    assert var_31 == '/tmp/file.txt'

import flutes.network as module_0

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = [var_4]
    var_6 = lambda self, chunk_size: var_5
    var_7 = 'download_warning_token'
    var_8 = 'yes'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = ()
    var_13 = [var_4]
    var_14 = lambda self, chunk_size: var_13
    var_15 = {}
    var_16 = {var_2: var_14, var_3: var_15}
    var_17 = [var_0, var_12, var_16]
    var_18 = 0
    assert var_18 == 2
    var_19 = 'MockSession'
    var_20 = ()
    var_21 = 'get'
    var_22 = 'MockRequests'
    var_23 = ()
    var_24 = 'Session'
    var_25 = None
    var_26 = 'MockOpen'
    var_27 = ()
    var_28 = '__enter__'
    var_29 = '__exit__'
    var_30 = 'write'
    var_31 = lambda self: self
    var_32 = lambda self, exc_type, exc_val, exc_tb: var_25
    var_33 = lambda self, data: var_25
    var_34 = {var_28: var_31, var_29: var_32, var_30: var_33}
    var_35 = [var_26, var_27, var_34]
    var_36 = 'https://drive.google.com/file/d/def456/view'
    var_37 = 'file.txt'
    var_38 = '/tmp'
    var_39 = module_0._download_from_google_drive(var_36, var_37, var_38, var_25)
    assert var_39 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = lambda self, chunk_size: var_6
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = 'MockSession'
    var_12 = ()
    var_13 = 'get'
    var_14 = 'MockRequests'
    var_15 = ()
    var_16 = 'Session'
    var_17 = None
    var_18 = 'MockOpen'
    var_19 = ()
    var_20 = '__enter__'
    var_21 = '__exit__'
    var_22 = 'write'
    var_23 = lambda self: self
    var_24 = lambda self, exc_type, exc_val, exc_tb: var_17
    var_25 = lambda self, data: var_17
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = [var_18, var_19, var_26]
    var_28 = []
    var_29 = 'MockProgress'
    var_30 = ()
    var_31 = 'update'
    var_32 = 'close'
    var_33 = lambda self, size: var_28.append(size)
    var_34 = lambda self: var_17
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = [var_29, var_30, var_35]
    var_37 = 'https://drive.google.com/file/d/ghi789/view'
    var_38 = 'file.txt'
    var_39 = '/tmp'
    var_40 = bool(var_28 == [6, 6])
    assert var_40 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_download_from_google_drive_with_valid_url_and_bar_fn. Retrieved 13/34 statements.
# Partially parsed test_download_from_google_drive_with_token_confirmation. Retrieved 16/38 statements.
# Partially parsed test_download_from_google_drive_without_bar_fn. Retrieved 12/23 statements.


def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = '/fake/path/file.txt'
    var_3 = 'https://drive.google.com/file/d/abc123/view'
    var_4 = 'file.txt'
    var_5 = '/fake/path'
    var_6 = 'https://docs.google.com/uc?export=download'
    var_7 = 'id'
    var_8 = 'abc123'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = 'wb'
    var_12 = 6

def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'confirm_value'
    var_2 = (var_0, var_1)
    var_3 = b'data'
    var_4 = '/path/file.txt'
    var_5 = 'https://drive.google.com/d/xyz456'
    var_6 = 'file.txt'
    var_7 = '/path'
    var_8 = 'https://docs.google.com/uc?export=download'
    var_9 = 'id'
    var_10 = 'xyz456'
    var_11 = {var_9: var_10}
    var_12 = True
    var_13 = 'confirm'
    var_14 = {var_9: var_10, var_13: var_1}
    var_15 = 4

import flutes.network as module_0

def test_case_0():
    var_0 = b'chunk'
    var_1 = '/some/path/file.bin'
    var_2 = 'https://drive.google.com/file/d/id789/'
    var_3 = 'file.bin'
    var_4 = '/some/path'
    var_5 = None
    var_6 = module_0._download_from_google_drive(var_2, var_3, var_4, var_5)
    assert var_6 == '/some/path/file.bin'
    var_7 = 'https://docs.google.com/uc?export=download'
    var_8 = 'id'
    var_9 = 'id789'
    var_10 = {var_8: var_9}
    var_11 = True

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/d/xyz456?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'xyz456'

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/d/fileId'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'fileId'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_download_from_google_drive_token_present. Retrieved 3/8 statements.
# Partially parsed test_download_from_google_drive_token_absent. Retrieved 3/8 statements.
# Failed to parse test_download_from_google_drive_token_empty_cookies.


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc123'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 'other_cookie'
    var_1 = 'value'
    var_2 = (var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_download_with_direct_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_direct_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_google_drive_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_google_drive_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_github_raw_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_without_save_dir_uses_temp_dir. Retrieved 2/4 statements.
# Partially parsed test_download_with_existing_file_skips_download. Retrieved 6/10 statements.
# Partially parsed test_download_with_extract_tar_file. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_zip_file. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 3/16 statements.


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.tar.gz'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'data.tar.gz'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/data.tar.gz'
    var_1 = '/tmp/test'
    var_2 = 'custom.tar.gz'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.tar.gz'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'abc123'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test'
    var_2 = 'file.zip'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'file.zip'
    var_6 = [var_5]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/data.zip?raw=true'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'data.zip'
    var_5 = [var_4]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/existing.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = {}
    var_7 = module_0.download(var_0, var_1, var_2, **var_6)

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = 'archive.tar.gz'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, var_3, **var_4)
    var_6 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = 'archive.zip'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, var_3, **var_4)
    var_6 = [var_2]

import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/large.bin'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'large.bin'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'https://example.com/file.bin'
    var_1 = '/tmp/test'
    var_2 = 'file.bin'
    var_3 = [var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_download_direct_url_with_default_filename. Retrieved 5/6 statements.
# Partially parsed test_download_direct_url_with_custom_filename. Retrieved 6/7 statements.
# Partially parsed test_download_google_drive_url. Retrieved 5/6 statements.
# Partially parsed test_download_github_raw_url. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 6/7 statements.
# Partially parsed test_download_with_extraction_tar. Retrieved 6/7 statements.
# Partially parsed test_download_with_extraction_zip. Retrieved 6/7 statements.
# Partially parsed test_download_existing_file_skips. Retrieved 6/9 statements.
# Partially parsed test_download_with_temporary_directory. Retrieved 3/5 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 3/19 statements.


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test_download'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'data.txt'
    var_5 = [var_4]
    var_6 = module_1.exists(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test_download'
    var_2 = 'custom.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.txt'
    var_6 = [var_5]
    var_7 = module_1.exists(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = '/tmp/test_download'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'abc123'
    var_5 = [var_4]
    var_6 = module_1.exists(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.txt?raw=true'
    var_1 = '/tmp/test_download'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = [var_4]
    var_6 = module_1.exists(var_3)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test_download'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'data.txt'
    var_6 = [var_5]
    var_7 = module_1.exists(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test_download'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.tar.gz'
    var_6 = [var_5]
    var_7 = module_1.exists(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test_download'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.zip'
    var_6 = [var_5]
    var_7 = module_1.exists(var_4)
    var_8 = bool(var_7)
    assert var_8 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test_download'
    var_2 = 'data.txt'
    var_3 = [var_2]
    var_4 = 'w'
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, **var_5)
    var_7 = module_1.exists(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)
    var_3 = module_1.exists(var_2)
    var_4 = bool(var_3)
    assert var_4 is True

def test_case_0():
    var_0 = 'https://example.com/data.txt'
    var_1 = '/tmp/test_download'
    var_2 = 'data.txt'
    var_3 = [var_2]



