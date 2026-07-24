####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_download_google_drive_url. Retrieved 5/6 statements.
# Partially parsed test_download_direct_url. Retrieved 5/6 statements.
# Partially parsed test_download_without_filename. Retrieved 5/6 statements.
# Partially parsed test_download_github_raw_url. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 6/7 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 6/7 statements.
# Partially parsed test_download_with_progress. Retrieved 5/6 statements.
# Partially parsed test_download_existing_file. Retrieved 6/9 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 3/17 statements.


import flutes.network as module_0


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1abc123def456/view'
    var_1 = '/tmp/test'
    var_2 = 'test_file'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'file.txt'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/data.tar.gz'
    var_1 = '/tmp/test'
    var_2 = False
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'data.tar.gz'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/script.py?raw=true'
    var_1 = '/tmp/test'
    var_2 = False
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'script.py'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = 'archive.tar.gz'
    var_3 = True
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, var_3, var_4, **var_5)
    var_7 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = 'archive.zip'
    var_3 = True
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, var_3, var_4, **var_5)
    var_7 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/large_file.bin'
    var_1 = '/tmp/test'
    var_2 = 'large_file.bin'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/existing.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = False
    var_6 = {}
    var_7 = module_0.download(var_0, var_1, var_2, progress=var_5, **var_6)

import genericpath as module_1


def test_case_0():
    var_0 = 'https://example.com/temp.txt'
    var_1 = False
    var_2 = {}
    var_3 = module_0.download(var_0, progress=var_1, **var_2)
    var_4 = module_1.exists(var_3)
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 'https://example.com/file.bin'
    var_1 = '/tmp/test'
    var_2 = 'file.bin'
    var_3 = [var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 21/35 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 29/50 statements.
# Partially parsed test_download_from_google_drive_with_progress. Retrieved 20/37 statements.



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
    var_17 = 'https://drive.google.com/file/d/abc123/view'
    var_18 = 'file.txt'
    var_19 = '/tmp'
    var_20 = None
    var_21 = module_0._download_from_google_drive(var_17, var_18, var_19, var_20)
    assert var_21 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'xyz'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'iter_content'
    var_6 = 'cookies'
    var_7 = b'data'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_5: var_9, var_6: var_2}
    var_11 = [var_3, var_4, var_10]
    var_12 = ()
    var_13 = b'final'
    var_14 = [var_13]
    var_15 = lambda chunk_size: var_14
    var_16 = {}
    var_17 = {var_5: var_15, var_6: var_16}
    var_18 = [var_3, var_12, var_17]
    var_19 = 0
    var_20 = 'MockSession'
    var_21 = ()
    var_22 = 'get'
    var_23 = 'MockRequests'
    var_24 = ()
    var_25 = 'Session'
    var_26 = 'https://drive.google.com/d/def456'
    var_27 = 'doc.pdf'
    var_28 = '/home/user'
    var_29 = None
    var_30 = module_0._download_from_google_drive(var_26, var_27, var_28, var_29)
    assert var_30 == '/home/user/doc.pdf'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'part1'
    var_5 = b'part2'
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
    var_17 = []
    var_18 = 'https://drive.google.com/d/ghi789'
    var_19 = 'image.png'
    var_20 = '/downloads'
    var_21 = bool(var_17 == [5, 5])
    assert var_21 is True


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'
    var_2 = 'https://drive.google.com/d/def456'
    var_3 = module_0._extract_google_drive_file_id(var_2)
    assert var_3 == 'def456'
    var_4 = 'https://drive.google.com/d/ghi789/extra'
    var_5 = module_0._extract_google_drive_file_id(var_4)
    assert var_5 == 'ghi789'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 26/50 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 28/58 statements.
# Partially parsed test_download_from_google_drive_with_progress_bar. Retrieved 38/66 statements.



def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = 'MockSession'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'MockRequests'
    var_12 = ()
    var_13 = 'Session'
    var_14 = 'MockOpen'
    var_15 = ()
    var_16 = '__enter__'
    var_17 = '__exit__'
    var_18 = 'write'
    var_19 = lambda self: self
    var_20 = None
    var_21 = lambda self, *args: var_20
    var_22 = lambda self, chunk: var_20
    var_23 = {var_16: var_19, var_17: var_21, var_18: var_22}
    var_24 = [var_14, var_15, var_23]
    var_25 = 'https://drive.google.com/file/d/abc123/view'
    var_26 = 'file.txt'
    var_27 = '/tmp'
    var_28 = module_0._download_from_google_drive(var_25, var_26, var_27, var_20)
    assert var_28 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'download_warning_token'
    var_5 = 'xyz'
    var_6 = b'chunk1'
    var_7 = [var_6]
    var_8 = 'MockSession'
    var_9 = ()
    var_10 = {}
    var_11 = [var_8, var_9, var_10]
    var_12 = 0
    var_13 = 'MockRequests'
    var_14 = ()
    var_15 = 'Session'
    var_16 = 'MockOpen'
    var_17 = ()
    var_18 = '__enter__'
    var_19 = '__exit__'
    var_20 = 'write'
    var_21 = lambda self: self
    var_22 = None
    var_23 = lambda self, *args: var_22
    var_24 = lambda self, chunk: var_22
    var_25 = {var_18: var_21, var_19: var_23, var_20: var_24}
    var_26 = [var_16, var_17, var_25]
    var_27 = 'https://drive.google.com/file/d/def456/view'
    var_28 = 'file.txt'
    var_29 = '/tmp'
    var_30 = module_0._download_from_google_drive(var_27, var_28, var_29, var_22)
    assert var_30 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = b'chunk3'
    var_7 = [var_4, var_5, var_6]
    var_8 = 'MockSession'
    var_9 = ()
    var_10 = {}
    var_11 = [var_8, var_9, var_10]
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = 'Session'
    var_15 = 'MockOpen'
    var_16 = ()
    var_17 = '__enter__'
    var_18 = '__exit__'
    var_19 = 'write'
    var_20 = lambda self: self
    var_21 = None
    var_22 = lambda self, *args: var_21
    var_23 = lambda self, chunk: var_21
    var_24 = {var_17: var_20, var_18: var_22, var_19: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = []
    var_27 = 'MockBar'
    var_28 = ()
    var_29 = 'update'
    var_30 = 'close'
    var_31 = lambda self, size: var_26.append(size)
    var_32 = lambda self: var_21
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = [var_27, var_28, var_33]
    var_35 = 'https://drive.google.com/file/d/ghi789/view'
    var_36 = 'file.txt'
    var_37 = '/tmp'
    var_38 = len(var_4)
    var_39 = len(var_5)
    var_40 = len(var_6)
    var_41 = [var_38, var_39, var_40]
    var_42 = bool(var_26 == var_41)
    assert var_42 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_download_with_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_google_drive_url. Retrieved 4/5 statements.
# Partially parsed test_download_github_raw_url. Retrieved 4/5 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_with_bar_fn. Retrieved 3/16 statements.
# Partially parsed test_download_existing_file. Retrieved 6/10 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 5/6 statements.
# Partially parsed test_download_with_unknown_compression. Retrieved 5/6 statements.
# Partially parsed test_download_without_save_dir. Retrieved 2/4 statements.
# Partially parsed test_download_with_kwargs. Retrieved 6/7 statements.



def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'custom.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.txt'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = '12345'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.txt?raw=true'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'file.txt'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'file.txt'
    var_3 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = {}
    var_7 = module_0.download(var_0, var_1, var_2, **var_6)


def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.tar.gz'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.zip'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/archive.rar'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.rar'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = {}
    var_2 = module_0.download(var_0, **var_1)


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = 'Downloading'
    var_4 = 'desc'
    var_5 = {var_4: var_3}
    var_6 = module_0.download(var_0, var_1, progress=var_2, **var_5)
    var_7 = 'file.txt'
    var_8 = [var_7]



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_download_without_progress. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress. Retrieved 3/16 statements.
# Partially parsed test_download_with_progress_no_total. Retrieved 3/16 statements.



def test_case_0():
    var_0 = 'http://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download(var_0, var_1, var_2, var_3)
    var_5 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = '/tmp'
    var_3 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = '/tmp'
    var_3 = [var_1]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_true. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'total'
    var_2 = 'refresh'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = 'file.txt'
    var_8 = '.'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_download_from_google_drive_success_without_token. Retrieved 30/49 statements.
# Partially parsed test_download_from_google_drive_success_with_token. Retrieved 37/63 statements.
# Partially parsed test_download_from_google_drive_with_progress_bar. Retrieved 36/58 statements.



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
    var_28 = 'https://drive.google.com/file/d/abc123/view'
    var_29 = 'file.txt'
    var_30 = '/tmp'
    var_31 = module_0._download_from_google_drive(var_28, var_29, var_30, var_23)
    assert var_31 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'value'
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
    var_6 = b'chunk3'
    var_7 = [var_4, var_5, var_6]
    var_8 = lambda self, chunk_size: var_7
    var_9 = {}
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = 'MockSession'
    var_13 = ()
    var_14 = 'get'
    var_15 = 'MockRequests'
    var_16 = ()
    var_17 = 'Session'
    var_18 = 'OpenMock'
    var_19 = ()
    var_20 = '__enter__'
    var_21 = '__exit__'
    var_22 = 'write'
    var_23 = lambda self: self
    var_24 = None
    var_25 = lambda self, *args: var_24
    var_26 = lambda self, chunk: var_24
    var_27 = {var_20: var_23, var_21: var_25, var_22: var_26}
    var_28 = [var_18, var_19, var_27]
    var_29 = []
    var_30 = False
    var_31 = 'https://drive.google.com/file/d/ghi789/view'
    var_32 = 'file.txt'
    var_33 = '/tmp'
    var_34 = len(var_29)
    assert var_34 == 3
    var_35 = len(var_4)
    var_36 = var_29[0]
    var_37 = bool(var_29[0] == var_35)
    assert var_37 is True
    var_38 = len(var_5)
    var_39 = var_29[1]
    var_40 = bool(var_29[1] == var_38)
    assert var_40 is True
    var_41 = len(var_6)
    var_42 = var_29[2]
    var_43 = bool(var_29[2] == var_41)
    assert var_43 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 22/42 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 23/43 statements.
# Partially parsed test_download_from_google_drive_with_progress_bar. Retrieved 27/51 statements.



def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = 'MockSession'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'MockRequests'
    var_12 = ()
    var_13 = {}
    var_14 = [var_11, var_12, var_13]
    var_15 = 'MockFile'
    var_16 = ()
    var_17 = 'write'
    var_18 = None
    var_19 = lambda self, data: var_18
    var_20 = {var_17: var_19}
    var_21 = [var_15, var_16, var_20]
    var_22 = 'https://drive.google.com/file/d/abc123/view'
    var_23 = 'file.txt'
    var_24 = '/tmp'
    var_25 = module_0._download_from_google_drive(var_22, var_23, var_24, var_18)
    assert var_25 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'download_warning_token'
    var_5 = 'xyz'
    var_6 = b'chunk1'
    var_7 = [var_6]
    var_8 = 'MockSession'
    var_9 = ()
    var_10 = {}
    var_11 = [var_8, var_9, var_10]
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = {}
    var_15 = [var_12, var_13, var_14]
    var_16 = 'MockFile'
    var_17 = ()
    var_18 = 'write'
    var_19 = None
    var_20 = lambda self, data: var_19
    var_21 = {var_18: var_20}
    var_22 = [var_16, var_17, var_21]
    var_23 = 'https://drive.google.com/d/def456'
    var_24 = 'data.zip'
    var_25 = '/downloads'
    var_26 = module_0._download_from_google_drive(var_23, var_24, var_25, var_19)
    assert var_26 == '/downloads/data.zip'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'data'
    var_5 = [var_4]
    var_6 = 'MockSession'
    var_7 = ()
    var_8 = {}
    var_9 = [var_6, var_7, var_8]
    var_10 = 'MockRequests'
    var_11 = ()
    var_12 = {}
    var_13 = [var_10, var_11, var_12]
    var_14 = 'MockFile'
    var_15 = ()
    var_16 = 'write'
    var_17 = None
    var_18 = lambda self, data: var_17
    var_19 = {var_16: var_18}
    var_20 = [var_14, var_15, var_19]
    var_21 = 'MockBar'
    var_22 = ()
    var_23 = 'update'
    var_24 = 'close'
    var_25 = lambda self, size: var_17
    var_26 = lambda self: var_17
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = [var_21, var_22, var_27]
    var_29 = 'https://drive.google.com/file/d/ghi789/edit'
    var_30 = 'image.png'
    var_31 = '/home/user'


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'


def test_case_0():
    var_0 = 'https://drive.google.com/d/def456'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'def456'


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/ghi789/edit?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'ghi789'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 27/49 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 29/54 statements.
# Partially parsed test_download_from_google_drive_no_bar. Retrieved 20/38 statements.


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
    var_17 = 'MockProgress'
    var_18 = ()
    var_19 = 'update'
    var_20 = 'close'
    var_21 = None
    var_22 = lambda x: var_21
    var_23 = lambda : var_21
    var_24 = {var_19: var_22, var_20: var_23}
    var_25 = [var_17, var_18, var_24]
    var_26 = 'https://drive.google.com/file/d/abc123/view'
    var_27 = 'file.txt'
    var_28 = '/tmp'


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse1'
    var_4 = ()
    var_5 = 'cookies'
    var_6 = 'iter_content'
    var_7 = []
    var_8 = lambda chunk_size: var_7
    var_9 = {var_5: var_2, var_6: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = 'MockResponse2'
    var_12 = ()
    var_13 = b'data'
    var_14 = [var_13]
    var_15 = lambda chunk_size: var_14
    var_16 = {}
    var_17 = {var_6: var_15, var_5: var_16}
    var_18 = [var_11, var_12, var_17]
    var_19 = 0
    assert var_19 == 2
    var_20 = 'MockSession'
    var_21 = ()
    var_22 = 'get'
    var_23 = 'MockRequests'
    var_24 = ()
    var_25 = 'Session'
    var_26 = 'https://drive.google.com/d/xyz456'
    var_27 = 'test.bin'
    var_28 = '.'
    var_29 = None
    var_30 = module_0._download_from_google_drive(var_26, var_27, var_28, var_29)


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'content'
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
    var_16 = 'https://drive.google.com/d/def789'
    var_17 = 'out.txt'
    var_18 = '/home/user'
    var_19 = None
    var_20 = module_0._download_from_google_drive(var_16, var_17, var_18, var_19)
    assert var_20 == '/home/user/out.txt'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 8/23 statements.



def test_case_0():
    var_0 = 'download_warning_123'
    var_1 = 'token_value'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = 'https://drive.google.com/file/d/123/view'
    var_5 = 'file.txt'
    var_6 = '/fake/path'
    var_7 = module_0._download_from_google_drive(var_4, var_5, var_6)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_download_without_progress. Retrieved 5/17 statements.
# Partially parsed test_download_with_progress. Retrieved 4/20 statements.
# Partially parsed test_download_progress_hook_initializes_progress. Retrieved 8/24 statements.
# Partially parsed test_download_progress_hook_updates_multiple_times. Retrieved 5/27 statements.
# Partially parsed test_download_progress_hook_with_unknown_total. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = None
    var_2 = 'http://example.com/file.txt'
    var_3 = 'test.txt'
    var_4 = None

def test_case_0():
    var_0 = 'test.txt'
    var_1 = None
    var_2 = 'http://example.com/file.txt'
    var_3 = 'test.txt'

def test_case_0():
    var_0 = 1
    var_1 = 1024
    var_2 = 2048
    var_3 = None
    var_4 = 'test.txt'
    var_5 = 'http://example.com/file.txt'
    var_6 = 'test.txt'
    var_7 = 1024

def test_case_0():
    var_0 = 0
    var_1 = 'http://example.com/file.txt'
    var_2 = 'test.txt'
    var_3 = 1024
    var_4 = 2048

def test_case_0():
    var_0 = 1
    var_1 = 1024
    var_2 = -1
    var_3 = None
    var_4 = 'test.txt'
    var_5 = 'http://example.com/file.txt'
    var_6 = 'test.txt'
    var_7 = 1024



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_download_without_progress_bar. Retrieved 5/17 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 4/20 statements.
# Partially parsed test_download_progress_hook_with_total_size. Retrieved 9/26 statements.
# Partially parsed test_download_progress_hook_multiple_updates. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = None
    var_2 = 'http://example.com/file.txt'
    var_3 = 'test.txt'
    var_4 = None

def test_case_0():
    var_0 = 'test.txt'
    var_1 = None
    var_2 = 'http://example.com/file.txt'
    var_3 = 'test.txt'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 1
    var_2 = 1024
    var_3 = 2048
    var_4 = 'http://example.com/file.txt'
    var_5 = 'test.txt'
    var_6 = 'total'
    var_7 = 2048
    var_8 = 1024

def test_case_0():
    var_0 = 0
    var_1 = 'http://example.com/file.txt'
    var_2 = 'test.txt'
    var_3 = 1024
    var_4 = 2048



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 21/33 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 29/48 statements.
# Partially parsed test_download_from_google_drive_with_progress. Retrieved 23/38 statements.



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
    var_17 = 'https://drive.google.com/file/d/abc123/view'
    var_18 = 'file.txt'
    var_19 = '/tmp'
    var_20 = None
    var_21 = module_0._download_from_google_drive(var_17, var_18, var_19, var_20)
    assert var_21 == '/tmp/file.txt'


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
    var_13 = b'chunk2'
    var_14 = [var_13]
    var_15 = lambda self, chunk_size: var_14
    var_16 = {}
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = [var_0, var_12, var_17]
    var_19 = 0
    var_20 = 'MockSession'
    var_21 = ()
    var_22 = 'get'
    var_23 = 'MockRequests'
    var_24 = ()
    var_25 = 'Session'
    var_26 = 'https://drive.google.com/d/def456'
    var_27 = 'data.bin'
    var_28 = '/home/user'
    var_29 = None
    var_30 = module_0._download_from_google_drive(var_26, var_27, var_28, var_29)
    assert var_30 == '/home/user/data.bin'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'a'
    var_5 = 32768
    var_6 = var_4 * var_5
    var_7 = b'b'
    var_8 = var_7 * var_5
    var_9 = [var_6, var_8]
    var_10 = lambda self, chunk_size: var_9
    var_11 = {}
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = 'MockSession'
    var_15 = ()
    var_16 = 'get'
    var_17 = 'MockRequests'
    var_18 = ()
    var_19 = 'Session'
    var_20 = []
    var_21 = 'https://drive.google.com/file/d/xyz789/'
    var_22 = 'large.txt'
    var_23 = '/var/tmp'
    var_24 = bool(var_20 == [32768, 32768])
    assert var_24 is True


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'


def test_case_0():
    var_0 = 'https://drive.google.com/d/def456'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'def456'


def test_case_0():
    var_0 = 'https://drive.google.com/d/ghi789/extra/path'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'ghi789'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/test_file_id/view'
    var_1 = 'test_file'
    var_2 = '/tmp'
    var_3 = 'download_warning_token'
    var_4 = 'test_token'
    var_5 = None
    assert var_5 == 'test_token'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 9/20 statements.



def test_case_0():
    var_0 = 'download_warning_123'
    var_1 = 'token_value'
    var_2 = b'chunk1'
    var_3 = b'chunk2'
    var_4 = [var_2, var_3]
    var_5 = 'https://drive.google.com/file/d/file_id/view'
    var_6 = 'file.txt'
    var_7 = '/fake/path'
    var_8 = module_0._download_from_google_drive(var_5, var_6, var_7)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_zipfile_extract_predicate_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 'test.zip'
    var_1 = 'http://example.com/test.zip'
    var_2 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_download_with_default_filename. Retrieved 2/12 statements.
# Partially parsed test_download_with_custom_filename. Retrieved 2/12 statements.
# Partially parsed test_download_google_drive_url. Retrieved 2/12 statements.
# Partially parsed test_download_existing_file_skips. Retrieved 4/15 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 3/13 statements.
# Partially parsed test_download_with_extraction. Retrieved 6/26 statements.
# Partially parsed test_download_github_raw_url. Retrieved 2/12 statements.
# Partially parsed test_download_no_save_dir_uses_temp. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'testfile.txt'
    var_1 = 'http://example.com/testfile.txt'

def test_case_0():
    var_0 = 'custom.txt'
    var_1 = 'http://example.com/testfile.txt'

def test_case_0():
    var_0 = 'file_id'
    var_1 = 'https://drive.google.com/file/d/file_id/view'

def test_case_0():
    var_0 = 'existing.txt'
    var_1 = 'content'
    var_2 = 'http://example.com/existing.txt'
    var_3 = 'existing.txt'

def test_case_0():
    var_0 = 'testfile.txt'
    var_1 = 'http://example.com/testfile.txt'
    var_2 = True

def test_case_0():
    var_0 = 'archive.tar.gz'
    var_1 = b'test content'
    var_2 = [var_1]
    var_3 = 'test.txt'
    var_4 = 'http://example.com/archive.tar.gz'
    var_5 = True
    var_6 = 'test.txt'

def test_case_0():
    var_0 = 'file.py'
    var_1 = 'https://github.com/user/repo/file.py?raw=true'


def test_case_0():
    var_0 = '/tmp'
    var_1 = 'testfile.txt'
    var_2 = [var_1]
    var_3 = 'http://example.com/testfile.txt'
    var_4 = {}
    var_5 = module_0.download(var_3, **var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 5/6 statements.



def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download(var_0, var_1, var_2, var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_false. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345/view'
    var_1 = 'test.txt'
    var_2 = b''
    var_3 = b''



# Parsed testcases at query #23
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = 'http://example.com'
    var_3 = 'file.txt'
    var_4 = '.'
    var_5 = module_0._download(var_2, var_3, var_4, var_1)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_download_without_progress. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress. Retrieved 13/20 statements.
# Partially parsed test_download_progress_hook_initialization. Retrieved 13/19 statements.
# Partially parsed test_download_progress_hook_with_total_size. Retrieved 13/19 statements.
# Partially parsed test_download_progress_hook_update. Retrieved 13/19 statements.



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
    var_3 = 'obj'
    var_4 = 'total'
    var_5 = 'refresh'
    var_6 = 'update'
    var_7 = 'close'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: var_8
    var_11 = lambda : var_8
    var_12 = {var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_11}
    var_13 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = 'obj'
    var_4 = 'total'
    var_5 = 'refresh'
    var_6 = 'update'
    var_7 = 'close'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: var_8
    var_11 = lambda : var_8
    var_12 = {var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_11}
    var_13 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = 'obj'
    var_4 = 'total'
    var_5 = 'refresh'
    var_6 = 'update'
    var_7 = 'close'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: var_8
    var_11 = lambda : var_8
    var_12 = {var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_11}
    var_13 = [var_1]

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = 'obj'
    var_4 = 'total'
    var_5 = 'refresh'
    var_6 = 'update'
    var_7 = 'close'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: var_8
    var_11 = lambda : var_8
    var_12 = {var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_11}
    var_13 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_download_google_drive_url. Retrieved 5/6 statements.
# Partially parsed test_download_direct_url. Retrieved 5/6 statements.
# Partially parsed test_download_without_filename. Retrieved 5/6 statements.
# Partially parsed test_download_without_save_dir. Retrieved 3/5 statements.
# Partially parsed test_download_existing_file. Retrieved 7/11 statements.
# Partially parsed test_download_with_progress. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 6/7 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 6/7 statements.
# Partially parsed test_download_github_raw_url. Retrieved 5/6 statements.
# Partially parsed test_download_custom_bar_fn. Retrieved 3/18 statements.



def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view'
    var_1 = '/tmp/test'
    var_2 = 'test_file.txt'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/file.zip'
    var_1 = '/tmp/test'
    var_2 = 'file.zip'
    var_3 = False
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/data.tar.gz'
    var_1 = '/tmp/test'
    var_2 = False
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'data.tar.gz'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = False
    var_2 = {}
    var_3 = module_0.download(var_0, progress=var_1, **var_2)


def test_case_0():
    var_0 = 'https://example.com/existing.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = False
    var_7 = {}
    var_8 = module_0.download(var_0, var_1, var_2, progress=var_6, **var_7)


def test_case_0():
    var_0 = 'https://example.com/large.bin'
    var_1 = '/tmp/test'
    var_2 = 'large.bin'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    var_6 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = 'archive.tar.gz'
    var_3 = True
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, var_3, var_4, **var_5)
    var_7 = [var_2]


def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = 'archive.zip'
    var_3 = True
    var_4 = False
    var_5 = {}
    var_6 = module_0.download(var_0, var_1, var_2, var_3, var_4, **var_5)
    var_7 = [var_2]


def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/script.py?raw=true'
    var_1 = '/tmp/test'
    var_2 = False
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'script.py'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'https://example.com/file.bin'
    var_1 = '/tmp/test'
    var_2 = 'file.bin'
    var_3 = [var_2]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 19/33 statements.
# Partially parsed test_download_from_google_drive_with_bar_fn. Retrieved 25/43 statements.
# Partially parsed test_download_from_google_drive_with_confirm_token. Retrieved 27/48 statements.



def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = 'Session'
    var_15 = 'https://drive.google.com/file/d/abc123/view'
    var_16 = 'file.txt'
    var_17 = '/tmp'
    var_18 = None
    var_19 = module_0._download_from_google_drive(var_15, var_16, var_17, var_18)
    assert var_19 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = 'Session'
    var_15 = 'MockProgress'
    var_16 = ()
    var_17 = 'update'
    var_18 = 'close'
    var_19 = None
    var_20 = lambda x: var_19
    var_21 = lambda : var_19
    var_22 = {var_17: var_20, var_18: var_21}
    var_23 = [var_15, var_16, var_22]
    var_24 = 'https://drive.google.com/file/d/abc123/view'
    var_25 = 'file.txt'
    var_26 = '/tmp'


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'confirm_value'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'cookies'
    var_6 = 'iter_content'
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_5: var_2, var_6: var_9}
    var_11 = [var_3, var_4, var_10]
    var_12 = ()
    var_13 = [var_7]
    var_14 = lambda chunk_size: var_13
    var_15 = {var_6: var_14}
    var_16 = [var_3, var_12, var_15]
    var_17 = 0
    assert var_17 == 2
    var_18 = 'MockSession'
    var_19 = ()
    var_20 = 'get'
    var_21 = 'MockRequests'
    var_22 = ()
    var_23 = 'Session'
    var_24 = 'https://drive.google.com/file/d/abc123/view'
    var_25 = 'file.txt'
    var_26 = '/tmp'
    var_27 = None
    var_28 = module_0._download_from_google_drive(var_24, var_25, var_26, var_27)
    assert var_28 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'
    var_2 = 'https://drive.google.com/file/d/xyz456/edit'
    var_3 = module_0._extract_google_drive_file_id(var_2)
    assert var_3 == 'xyz456'
    var_4 = 'https://drive.google.com/drive/folders/def789'
    var_5 = module_0._extract_google_drive_file_id(var_4)
    assert var_5 == 'def789'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 6/10 statements.
# Partially parsed test_download_from_google_drive_without_token. Retrieved 4/8 statements.
# Partially parsed test_download_from_google_drive_with_irrelevant_cookies. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'cookies'
    var_2 = 'download_warning_token'
    var_3 = 'abc123'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'obj'
    var_1 = 'cookies'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'obj'
    var_1 = 'cookies'
    var_2 = 'session'
    var_3 = 'user'
    var_4 = 'xyz'
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_progress_close_called_when_bar_fn_provided. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/1xO1uCb0KqB8l5ZqQw8XyVz2b3c4d5e6f/view'
    var_1 = 'test.txt'
    var_2 = b'chunk1'
    var_3 = b'chunk2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_false. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/12345/view'
    var_1 = 'test.txt'
    var_2 = b''
    var_3 = bool(not b'')
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__download_from_google_drive_success_with_bar. Retrieved 33/61 statements.
# Partially parsed test__download_from_google_drive_success_without_bar. Retrieved 27/51 statements.
# Partially parsed test__download_from_google_drive_with_token. Retrieved 32/65 statements.


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = 'MockSession'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'MockRequests'
    var_12 = ()
    var_13 = 'Session'
    var_14 = 'MockOS'
    var_15 = ()
    var_16 = 'path'
    var_17 = 'MockPath'
    var_18 = ()
    var_19 = 'join'
    var_20 = lambda path, filename: f'{path}/{filename}'
    var_21 = {var_19: var_20}
    var_22 = [var_17, var_18, var_21]
    var_23 = 'MockBar'
    var_24 = ()
    var_25 = 'update'
    var_26 = 'close'
    var_27 = None
    var_28 = lambda x: var_27
    var_29 = lambda : var_27
    var_30 = {var_25: var_28, var_26: var_29}
    var_31 = [var_23, var_24, var_30]
    var_32 = 'https://drive.google.com/file/d/abc123/view'
    var_33 = 'file.txt'
    var_34 = '/tmp'
    var_35 = 'requests'
    var_36 = 'os'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = b'chunk1'
    var_5 = b'chunk2'
    var_6 = [var_4, var_5]
    var_7 = 'MockSession'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'MockRequests'
    var_12 = ()
    var_13 = 'Session'
    var_14 = 'MockOS'
    var_15 = ()
    var_16 = 'path'
    var_17 = 'MockPath'
    var_18 = ()
    var_19 = 'join'
    var_20 = lambda path, filename: f'{path}/{filename}'
    var_21 = {var_19: var_20}
    var_22 = [var_17, var_18, var_21]
    var_23 = 'https://drive.google.com/file/d/abc123/view'
    var_24 = 'file.txt'
    var_25 = '/tmp'
    var_26 = None
    var_27 = module_0._download_from_google_drive(var_23, var_24, var_25, var_26)
    assert var_27 == '/tmp/file.txt'
    var_28 = 'requests'
    var_29 = 'os'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'download_warning_token'
    var_5 = 'token_value'
    var_6 = ()
    var_7 = {}
    var_8 = [var_0, var_6, var_7]
    var_9 = b'chunk1'
    var_10 = b'chunk2'
    var_11 = [var_9, var_10]
    var_12 = 'MockSession'
    var_13 = ()
    var_14 = {}
    var_15 = [var_12, var_13, var_14]
    var_16 = 0
    assert var_16 == 2
    var_17 = 'MockRequests'
    var_18 = ()
    var_19 = 'Session'
    var_20 = 'MockOS'
    var_21 = ()
    var_22 = 'path'
    var_23 = 'MockPath'
    var_24 = ()
    var_25 = 'join'
    var_26 = lambda path, filename: f'{path}/{filename}'
    var_27 = {var_25: var_26}
    var_28 = [var_23, var_24, var_27]
    var_29 = 'https://drive.google.com/file/d/abc123/view'
    var_30 = 'file.txt'
    var_31 = '/tmp'
    var_32 = None
    var_33 = module_0._download_from_google_drive(var_29, var_30, var_31, var_32)
    assert var_33 == '/tmp/file.txt'
    var_34 = 'requests'
    var_35 = 'os'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 6/18 statements.



def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'some_token'
    var_2 = 'https://drive.google.com/file/d/12345/view'
    var_3 = 'file.txt'
    var_4 = '/fake/path'
    var_5 = module_0._download_from_google_drive(var_2, var_3, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_download_without_progress. Retrieved 5/6 statements.
# Partially parsed test_download_with_progress. Retrieved 3/16 statements.
# Partially parsed test_download_progress_hook_initialization. Retrieved 3/16 statements.
# Partially parsed test_download_progress_total_set. Retrieved 3/16 statements.
# Partially parsed test_download_progress_update_called. Retrieved 3/16 statements.
# Partially parsed test_download_progress_close_called. Retrieved 3/17 statements.



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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 27/45 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 26/47 statements.
# Partially parsed test_download_from_google_drive_no_bar. Retrieved 20/34 statements.


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
    var_17 = 'MockBar'
    var_18 = ()
    var_19 = 'update'
    var_20 = 'close'
    var_21 = None
    var_22 = lambda x: var_21
    var_23 = lambda : var_21
    var_24 = {var_19: var_22, var_20: var_23}
    var_25 = [var_17, var_18, var_24]
    var_26 = 'https://drive.google.com/file/d/abc123/view'
    var_27 = 'file.txt'
    var_28 = '/tmp'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'cookies'
    var_3 = 'download_warning_token'
    var_4 = 'yes'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = ()
    var_9 = 'iter_content'
    var_10 = b'data'
    var_11 = [var_10]
    var_12 = lambda chunk_size: var_11
    var_13 = {}
    var_14 = {var_9: var_12, var_2: var_13}
    var_15 = [var_0, var_8, var_14]
    var_16 = 0
    assert var_16 == 2
    var_17 = 'MockSession'
    var_18 = ()
    var_19 = 'get'
    var_20 = 'MockRequests'
    var_21 = ()
    var_22 = 'Session'
    var_23 = 'https://drive.google.com/d/def456/'
    var_24 = 'test.bin'
    var_25 = '.'
    var_26 = None
    var_27 = module_0._download_from_google_drive(var_23, var_24, var_25, var_26)
    assert var_27 == './test.bin'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = 'cookies'
    var_4 = b'content'
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
    var_16 = 'https://drive.google.com/d/ghi789'
    var_17 = 'out.dat'
    var_18 = '/home/user'
    var_19 = None
    var_20 = module_0._download_from_google_drive(var_16, var_17, var_18, var_19)
    assert var_20 == '/home/user/out.dat'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = None
    assert var_2 is None
    var_3 = 0
    var_4 = lambda count, block_size, total_size: var_0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_download_with_direct_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_direct_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_google_drive_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_with_google_drive_url_and_custom_filename. Retrieved 5/6 statements.
# Partially parsed test_download_with_github_raw_url_and_default_filename. Retrieved 4/5 statements.
# Partially parsed test_download_without_save_dir_uses_temp_dir. Retrieved 3/5 statements.
# Partially parsed test_download_with_existing_file_skips_download. Retrieved 6/10 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 5/6 statements.
# Partially parsed test_download_with_custom_bar_fn. Retrieved 5/21 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 5/6 statements.
# Partially parsed test_download_with_extract_unknown_type. Retrieved 5/6 statements.



def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'custom.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.txt'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/file_id/view'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file_id'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/file_id/view'
    var_1 = '/tmp/test'
    var_2 = 'custom.txt'
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, var_2, **var_3)
    var_5 = 'custom.txt'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://github.com/user/repo/raw/main/file.txt?raw=true'
    var_1 = '/tmp/test'
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)
    var_4 = 'file.txt'
    var_5 = [var_4]


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = None
    var_2 = {}
    var_3 = module_0.download(var_0, var_1, **var_2)


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = 'existing.txt'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = {}
    var_7 = module_0.download(var_0, var_1, var_2, **var_6)


def test_case_0():
    var_0 = 'https://example.com/file.txt'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, progress=var_2, **var_3)
    var_5 = 'file.txt'
    var_6 = [var_5]

def test_case_0():
    var_0 = []
    var_1 = 'https://example.com/file.txt'
    var_2 = '/tmp/test'
    var_3 = 'file.txt'
    var_4 = [var_3]
    var_5 = len(var_0)
    var_6 = bool(var_5 > 0)
    assert var_6 is True


def test_case_0():
    var_0 = 'https://example.com/archive.tar.gz'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.tar.gz'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/archive.zip'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'archive.zip'
    var_6 = [var_5]


def test_case_0():
    var_0 = 'https://example.com/unknown.xyz'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    var_4 = module_0.download(var_0, var_1, extract=var_2, **var_3)
    var_5 = 'unknown.xyz'
    var_6 = [var_5]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_download_with_direct_url_and_default_filename. Retrieved 3/16 statements.
# Partially parsed test_download_with_direct_url_and_custom_filename. Retrieved 3/16 statements.
# Partially parsed test_download_with_google_drive_url. Retrieved 4/17 statements.
# Partially parsed test_download_with_existing_file_skips_download. Retrieved 4/15 statements.
# Partially parsed test_download_with_progress_bar. Retrieved 4/21 statements.
# Partially parsed test_download_with_extract_tar. Retrieved 7/27 statements.
# Partially parsed test_download_with_extract_zip. Retrieved 7/24 statements.
# Partially parsed test_download_with_unknown_compression_warns. Retrieved 7/23 statements.
# Partially parsed test_download_with_github_raw_url_removes_suffix. Retrieved 3/16 statements.
# Partially parsed test_download_with_temporary_directory. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'http://example.com/test.txt'
    var_2 = 'test.txt'

def test_case_0():
    var_0 = 'custom.txt'
    var_1 = 'http://example.com/test.txt'
    var_2 = 'custom.txt'

def test_case_0():
    var_0 = b'chunk1'
    var_1 = b'chunk2'
    var_2 = 'https://drive.google.com/file/d/abc123/view'
    var_3 = 'abc123'

def test_case_0():
    var_0 = 'existing.txt'
    var_1 = 'existing content'
    var_2 = 'http://example.com/existing.txt'
    var_3 = 'existing.txt'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'http://example.com/test.txt'
    var_2 = True
    var_3 = 'test.txt'

def test_case_0():
    var_0 = 'archive.tar.gz'
    var_1 = 'file.txt'
    var_2 = b'content'
    var_3 = [var_2]
    var_4 = 'rb'
    var_5 = 'http://example.com/archive.tar.gz'
    var_6 = True
    var_7 = 'file.txt'

def test_case_0():
    var_0 = 'archive.zip'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'rb'
    var_4 = 'http://example.com/archive.zip'
    var_5 = True
    var_6 = 'file.txt'

def test_case_0():
    var_0 = 'unknown.xyz'
    var_1 = 'content'
    var_2 = 'rb'
    var_3 = 'http://example.com/unknown.xyz'
    var_4 = True
    var_5 = 'Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported'
    var_6 = 'warning'

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'http://github.com/user/repo/file.txt?raw=true'
    var_2 = 'file.txt'


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'http://example.com/test.txt'
    var_2 = None
    var_3 = {}
    var_4 = module_0.download(var_1, var_2, **var_3)



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_download_from_google_drive_with_bar_fn. Retrieved 22/33 statements.
# Partially parsed test_download_from_google_drive_without_bar_fn. Retrieved 16/22 statements.
# Partially parsed test_download_from_google_drive_with_confirm_token. Retrieved 24/32 statements.
# Partially parsed test_download_from_google_drive_extract_file_id. Retrieved 15/21 statements.


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'MockBar'
    var_13 = ()
    var_14 = 'update'
    var_15 = 'close'
    var_16 = None
    var_17 = lambda self, size: var_16
    var_18 = lambda self: var_16
    var_19 = {var_14: var_17, var_15: var_18}
    var_20 = [var_12, var_13, var_19]
    var_21 = 'https://drive.google.com/file/d/abc123/view'
    var_22 = 'test.txt'
    var_23 = '/tmp'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'https://drive.google.com/file/d/abc123/view'
    var_13 = 'test.txt'
    var_14 = '/tmp'
    var_15 = None
    var_16 = module_0._download_from_google_drive(var_12, var_13, var_14, var_15)
    assert var_16 == '/tmp/test.txt'


def test_case_0():
    var_0 = 'download_warning_token'
    var_1 = 'abc'
    var_2 = {var_0: var_1}
    var_3 = 'MockResponse'
    var_4 = ()
    var_5 = 'cookies'
    var_6 = 'iter_content'
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_5: var_2, var_6: var_9}
    var_11 = [var_3, var_4, var_10]
    var_12 = ()
    var_13 = [var_7]
    var_14 = lambda chunk_size: var_13
    var_15 = {var_6: var_14}
    var_16 = [var_3, var_12, var_15]
    var_17 = 'MockSession'
    var_18 = ()
    var_19 = 'get'
    var_20 = 'confirm'
    var_21 = 'https://drive.google.com/file/d/abc123/view'
    var_22 = 'test.txt'
    var_23 = '/tmp'
    var_24 = None
    var_25 = module_0._download_from_google_drive(var_21, var_22, var_23, var_24)
    assert var_25 == '/tmp/test.txt'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = [var_3]
    var_5 = lambda chunk_size: var_4
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'MockSession'
    var_9 = ()
    var_10 = 'get'
    var_11 = 'https://drive.google.com/file/d/xyz789/view'
    var_12 = 'test.txt'
    var_13 = '/tmp'
    var_14 = None
    var_15 = module_0._download_from_google_drive(var_11, var_12, var_13, var_14)
    assert var_15 == '/tmp/test.txt'



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = module_0._download(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_download_with_progress. Retrieved 3/15 statements.
# Partially parsed test_download_progress_total_set. Retrieved 3/15 statements.
# Partially parsed test_download_with_progress_multiple_updates. Retrieved 3/16 statements.



def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = module_0._download(var_0, var_1, var_2, var_3)
    assert var_4 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'

def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'


def test_case_0():
    var_0 = 'http://example.com/data.bin'
    var_1 = 'data.bin'
    var_2 = '/home/user'
    var_3 = None
    var_4 = module_0._download(var_0, var_1, var_2, var_3)
    assert var_4 == '/home/user/data.bin'

def test_case_0():
    var_0 = 'http://example.com/large.bin'
    var_1 = 'large.bin'
    var_2 = '/tmp'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_download_from_google_drive_with_token. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/test_file_id/view'
    var_1 = 'test_file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = 'download_warning_token'
    var_5 = 'test_token'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_progress_is_none_when_bar_fn_is_none. Retrieved 10/16 statements.



def test_case_0():
    var_0 = 'http://example.com/file.txt'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = '/tmp/file.txt'
    var_5 = module_0._download(var_0, var_1, var_2, var_3)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True
    var_7 = [var_1]
    var_8 = None
    var_9 = 'progress'
    var_10 = locals()
    var_11 = var_9 not in var_10



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_progress_not_none_when_bar_fn_provided. Retrieved 15/30 statements.


def test_case_0():
    var_0 = 'obj'
    var_1 = 'total'
    var_2 = 'refresh'
    var_3 = 'update'
    var_4 = 'close'
    var_5 = None
    var_6 = lambda : var_5
    var_7 = lambda x: var_5
    var_8 = lambda : var_5
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = None
    var_11 = 0
    var_12 = 1
    var_13 = 1024
    var_14 = 2048
    var_15 = bool(var_10 is not None)
    assert var_15 is True



# Parsed testcases at query #21
#--------------------------





def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'file.txt'
    var_2 = '/tmp'
    var_3 = None
    var_4 = lambda : var_3
    var_5 = module_0._download(var_0, var_1, var_2, var_4)
    var_6 = module_1.exists(var_5)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_download_from_google_drive_success. Retrieved 28/47 statements.
# Partially parsed test_download_from_google_drive_with_bar_fn. Retrieved 34/57 statements.
# Partially parsed test_download_from_google_drive_with_token. Retrieved 38/59 statements.



def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = 'Session'
    var_15 = 'OpenMock'
    var_16 = ()
    var_17 = '__enter__'
    var_18 = '__exit__'
    var_19 = 'write'
    var_20 = lambda self: self
    var_21 = None
    var_22 = lambda self, *args: var_21
    var_23 = lambda self, chunk: var_21
    var_24 = {var_17: var_20, var_18: var_22, var_19: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'https://drive.google.com/file/d/abc123/view'
    var_27 = 'file.txt'
    var_28 = '/tmp'
    var_29 = module_0._download_from_google_drive(var_26, var_27, var_28, var_21)
    assert var_29 == '/tmp/file.txt'

def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'iter_content'
    var_3 = b'chunk1'
    var_4 = b'chunk2'
    var_5 = [var_3, var_4]
    var_6 = lambda chunk_size: var_5
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = 'MockSession'
    var_10 = ()
    var_11 = 'get'
    var_12 = 'MockRequests'
    var_13 = ()
    var_14 = 'Session'
    var_15 = 'OpenMock'
    var_16 = ()
    var_17 = '__enter__'
    var_18 = '__exit__'
    var_19 = 'write'
    var_20 = lambda self: self
    var_21 = None
    var_22 = lambda self, *args: var_21
    var_23 = lambda self, chunk: var_21
    var_24 = {var_17: var_20, var_18: var_22, var_19: var_23}
    var_25 = [var_15, var_16, var_24]
    var_26 = 'MockProgress'
    var_27 = ()
    var_28 = 'update'
    var_29 = 'close'
    var_30 = lambda self, size: var_21
    var_31 = lambda self: var_21
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = [var_26, var_27, var_32]
    var_34 = 'https://drive.google.com/file/d/abc123/view'
    var_35 = 'file.txt'
    var_36 = '/tmp'


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'cookies'
    var_3 = 'iter_content'
    var_4 = 'download_warning_token'
    var_5 = 'token123'
    var_6 = {var_4: var_5}
    var_7 = b'chunk1'
    var_8 = [var_7]
    var_9 = lambda chunk_size: var_8
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = ()
    var_13 = [var_7]
    var_14 = lambda chunk_size: var_13
    var_15 = {var_3: var_14}
    var_16 = [var_0, var_12, var_15]
    var_17 = 'MockSession'
    var_18 = ()
    var_19 = 'get'
    var_20 = 'id'
    var_21 = 'abc123'
    var_22 = 'confirm'
    var_23 = 'MockRequests'
    var_24 = ()
    var_25 = 'Session'
    var_26 = 'OpenMock'
    var_27 = ()
    var_28 = '__enter__'
    var_29 = '__exit__'
    var_30 = 'write'
    var_31 = lambda self: self
    var_32 = None
    var_33 = lambda self, *args: var_32
    var_34 = lambda self, chunk: var_32
    var_35 = {var_28: var_31, var_29: var_33, var_30: var_34}
    var_36 = [var_26, var_27, var_35]
    var_37 = 'https://drive.google.com/file/d/abc123/view'
    var_38 = 'file.txt'
    var_39 = '/tmp'
    var_40 = module_0._download_from_google_drive(var_37, var_38, var_39, var_32)
    assert var_40 == '/tmp/file.txt'


def test_case_0():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'abc123'
    var_2 = 'https://drive.google.com/file/d/xyz456/edit'
    var_3 = module_0._extract_google_drive_file_id(var_2)
    assert var_3 == 'xyz456'
    var_4 = 'https://drive.google.com/drive/folders/def789'
    var_5 = module_0._extract_google_drive_file_id(var_4)
    assert var_5 == 'def789'



