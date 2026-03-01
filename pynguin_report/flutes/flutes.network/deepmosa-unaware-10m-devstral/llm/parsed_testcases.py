####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test2.txt'
    var_3 = 'https://example.com/test.zip'
    var_4 = True
    var_5 = 'https://example.com/test3.txt'
    var_6 = 'https://drive.google.com/file/d/test_file_id/view'
    var_7 = 'https://example.com/test4.txt'



# Parsed testcases at query #2
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = True
    var_4 = True
    var_5 = 'https://example.com/test_file.zip'
    var_6 = 'test_file.zip'
    var_7 = 'https://drive.google.com/file/d/test_file_id/view'
    var_8 = 'test_file_id'
    var_9 = None
    var_10 = module_0.download(var_7, var_9, var_8)
    var_11 = module_1.exists(var_10)
    var_12 = module_2.dirname(var_10)



# Parsed testcases at query #3
#--------------------------


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = 'test_file.txt'
    var_4 = 'custom_test_file.txt'
    var_5 = True
    var_6 = 'test_file.txt'
    var_7 = 'https://drive.google.com/file/d/test_file_id/view'
    var_8 = 'test_file_id'
    var_9 = 'download_warning_test_file_id'
    var_10 = 'test_token'
    var_11 = (var_9, var_10)
    var_12 = b'test data'
    var_13 = 'https://example.com/test_file.zip'
    var_14 = 'test_file.zip'
    var_15 = None
    var_16 = True
    var_17 = None
    var_18 = module_0.download(var_0)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://drive.google.com/file/d/abc123/view'
    var_3 = 'gdrive_file.txt'
    var_4 = 'abc123'
    var_5 = 'https://example.com/test_archive.zip'
    var_6 = 'test_archive.zip'
    var_7 = True
    var_8 = 'extracted_file.txt'
    var_9 = 'https://example.com/large_file.txt'
    var_10 = 'large_file.txt'
    var_11 = 'custom_file.txt'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test2.txt'
    var_3 = 'test2.txt'
    var_4 = 'https://example.com/test3.txt'
    var_5 = 'test3.txt'
    var_6 = True
    var_7 = 'https://example.com/test4.txt'
    var_8 = 'test4.txt'
    var_9 = 'https://example.com/test.zip'
    var_10 = 'test.zip'
    var_11 = 'https://drive.google.com/file/d/test_id/view'
    var_12 = 'test_id'
    var_13 = 'https://example.com/nonexistent.txt'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = None
    var_3 = None
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = b'test content'
    var_7 = [var_6]
    var_8 = 'test.zip'
    var_9 = None
    var_10 = True
    var_11 = 'test.txt'
    var_12 = None
    var_13 = 'https://raw.githubusercontent.com/user/repo/main/test.txt?raw=true'
    var_14 = 'test.txt'
    var_15 = None



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'https://example.com/test.txt'
    var_4 = 'test.txt'
    var_5 = True
    var_6 = 'https://example.com/test.txt'
    var_7 = 'test.txt'
    var_8 = 'https://example.com/test.zip'
    var_9 = 'test.zip'
    var_10 = 'https://drive.google.com/file/d/123456789/view'
    var_11 = '123456789'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'



# Parsed testcases at query #8
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'extracted_file.txt'
    var_7 = 'https://example.com/test.tar.gz'
    var_8 = 'test.tar.gz'
    var_9 = 'https://drive.google.com/file/d/123456789/view'
    var_10 = '123456789'
    var_11 = None
    var_12 = module_0.download(var_0, var_11, var_1)
    var_13 = module_1.exists(var_12)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'test_file.txt'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/test_file_id/view'
    var_5 = 'https://example.com/test_file.zip'
    var_6 = 'test_file.zip'
    var_7 = 'extracted_file'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'extracted_file'
    var_8 = 'https://example.com/test.txt'
    var_9 = 'test.txt'
    var_10 = 'https://drive.google.com/file/d/test_id/view'
    var_11 = 'test_id'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'



# Parsed testcases at query #11
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = True
    var_3 = 'https://example.com/test_file.zip'
    var_4 = 'test_file.zip'
    var_5 = 'https://example.com/test_file.tar.gz'
    var_6 = 'test_file.tar.gz'
    var_7 = 'https://drive.google.com/file/d/test_file_id/view'
    var_8 = 'test_file.txt'
    var_9 = None
    var_10 = module_0.download(var_0, var_9, var_1)
    var_11 = module_1.exists(var_10)
    var_12 = module_2.basename(var_10)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = 'https://drive.google.com/file/d/test_file_id/view'
    var_4 = b'test content'
    var_5 = 'https://example.com/test_file.zip'
    var_6 = 'test_file.zip'
    var_7 = None
    var_8 = True
    var_9 = 'https://example.com/test_file.tar.gz'
    var_10 = 'test_file.tar.gz'
    var_11 = None
    var_12 = True
    var_13 = 'r'
    var_14 = None
    var_15 = True
    var_16 = None
    var_17 = True



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = None
    var_4 = {}
    var_5 = True
    var_6 = 'https://drive.google.com/file/d/test_id/view'
    var_7 = 'test_id'
    var_8 = b'test content'
    var_9 = 'https://example.com/test.zip'
    var_10 = 'test.zip'
    var_11 = {}
    var_12 = True
    var_13 = 'existing.txt'
    var_14 = 'existing content'
    var_15 = 'https://example.com/existing.txt'
    var_16 = 'existing.txt'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = 'test_id'



# Parsed testcases at query #15
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/test_id/view'
    var_4 = 'test_id'
    var_5 = 'https://example.com/test.zip'
    var_6 = 'test.zip'
    var_7 = 'test'
    var_8 = module_0.download(var_0)
    var_9 = module_1.exists(var_8)
    var_10 = module_2.dirname(var_8)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = None
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/test_file_id/view'
    var_6 = b'test content'
    var_7 = 'https://example.com/test_file.zip'
    var_8 = 'test_file.zip'
    var_9 = None
    var_10 = True
    var_11 = 'test_file.txt'
    var_12 = None



# Parsed testcases at query #17
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://drive.google.com/file/d/123456789/view'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'test'
    var_7 = 'https://example.com/test.tar.gz'
    var_8 = 'test.tar.gz'
    var_9 = 'test.txt'
    var_10 = None
    var_11 = module_0.download(var_0, var_10, var_1)
    var_12 = module_1.exists(var_11)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.txt'
    var_5 = 'test.txt'
    var_6 = True
    var_7 = 'https://example.com/test.txt'
    var_8 = 'test.txt'
    var_9 = 'https://example.com/test.zip'
    var_10 = 'test.zip'
    var_11 = 'https://drive.google.com/file/d/test_id/view'
    var_12 = 'test_id'



# Parsed testcases at query #19
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = 'test_id'
    var_7 = module_0.download(var_5)
    var_8 = module_1.exists(var_7)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = {}
    var_3 = None
    var_4 = {}
    var_5 = True
    var_6 = 'https://drive.google.com/file/d/test_file_id/view'
    var_7 = b'test content'
    var_8 = True
    var_9 = True
    var_10 = True
    var_11 = 'Unknown compression type. Only .tar.gz, .tar.bz2, .tar, and .zip are supported'
    var_12 = 'warning'



# Parsed testcases at query #21
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://drive.google.com/file/d/123456789/view'
    var_3 = '123456789'
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'test_file.zip'
    var_6 = True
    var_7 = 'extracted_file'
    var_8 = 'https://example.com/test_file.tar.gz'
    var_9 = 'test_file.tar.gz'
    var_10 = 'https://example.com/test_file.txt'
    var_11 = 'test_file.txt'
    var_12 = 'https://example.com/test_file.txt'
    var_13 = 'test_file.txt'
    var_14 = 'https://example.com/test_file.txt'
    var_15 = 'test_file.txt'
    var_16 = 'https://example.com/test_file.txt'
    var_17 = module_0.download(var_16)
    var_18 = module_1.exists(var_17)



# Parsed testcases at query #22
#--------------------------


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = None
    var_4 = {}
    var_5 = True
    var_6 = {}
    var_7 = True
    var_8 = 'https://drive.google.com/file/d/test_id/view'
    var_9 = 'download_warning_test_id'
    var_10 = 'test_token'
    var_11 = (var_9, var_10)
    var_12 = b'test content'
    var_13 = True
    var_14 = True
    var_15 = {}
    var_16 = module_0.download(var_0, filename=var_1)
    var_17 = 'test.txt'
    var_18 = {}
    var_19 = 'https://github.com/user/repo/raw/main/test.txt?raw=true'
    var_20 = 'test.txt'
    var_21 = {}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://drive.google.com/file/d/test_file_id/view'
    var_3 = 'test_file_id'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'extracted_file'
    var_8 = 'https://example.com/test.txt'
    var_9 = 'test.txt'
    var_10 = 'https://example.com/test.txt'
    var_11 = 'test.txt'



# Parsed testcases at query #24
#--------------------------


import flutes.network as module_0

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = None
    var_3 = None
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = b'test content'
    var_7 = 'test.zip'
    var_8 = None
    var_9 = True
    var_10 = None
    var_11 = module_0.download(var_0, filename=var_1)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'test_file.txt'
    var_3 = True
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'test_file.zip'
    var_6 = 'https://drive.google.com/file/d/test_file_id/view'
    var_7 = 'test_file_id'
    var_8 = 'https://example.com/non_existent_file.txt'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/test_id/view'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'test'
    var_7 = 'https://example.com/nonexistent.txt'



# Parsed testcases at query #2
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'custom.txt'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = 'test_id'
    var_7 = module_0.download(var_0)
    var_8 = module_1.exists(var_7)
    var_9 = module_2.dirname(var_7)
    var_10 = module_1.getmtime(var_7)
    var_11 = module_1.getmtime(var_7)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = True
    var_6 = 'https://drive.google.com/file/d/test_id/view'
    var_7 = 'test'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = True
    var_3 = 'https://example.com/test_file.zip'
    var_4 = 'test_file.zip'
    var_5 = 'extracted_file.txt'
    var_6 = 'https://example.com/test_file.tar.gz'
    var_7 = 'test_file.tar.gz'
    var_8 = 'https://drive.google.com/file/d/123456789/view'
    var_9 = 'test_file.txt'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = b'PK\x03\x04...'
    var_4 = 'https://drive.google.com/file/d/test_id/view'
    var_5 = 'test_gdrive.txt'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://example.com/test_file.txt'
    var_3 = 'https://example.com/test_file.txt'
    var_4 = 'test_file.txt'
    var_5 = True
    var_6 = 'https://example.com/test_file.zip'
    var_7 = 'test_file.zip'
    var_8 = 'https://example.com/test_file.tar.gz'
    var_9 = 'test_file.tar.gz'
    var_10 = 'https://drive.google.com/file/d/test_file_id/view'
    var_11 = 'test_file_id'
    var_12 = 'https://example.com/test_file.txt'
    var_13 = 'test_file.txt'



# Parsed testcases at query #7
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'https://example.com/test.txt'
    var_4 = 'test.txt'
    var_5 = True
    var_6 = 'https://example.com/test.txt'
    var_7 = 'test.txt'
    var_8 = 'https://example.com/test.zip'
    var_9 = 'test.zip'
    var_10 = 'https://drive.google.com/file/d/test_id/view'
    var_11 = 'test_id'
    var_12 = 'https://example.com/test.txt'
    var_13 = None
    var_14 = module_0.download(var_12, var_13)
    var_15 = module_1.exists(var_14)
    var_16 = module_2.basename(var_14)
    assert var_16 == 'test.txt'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://drive.google.com/file/d/test_file_id/view'
    var_3 = 'test_file_id'
    var_4 = 'https://example.com/test_archive.zip'
    var_5 = 'test_archive.zip'
    var_6 = True
    var_7 = 'https://example.com/test_archive.tar.gz'
    var_8 = 'test_archive.tar.gz'
    var_9 = 'https://example.com/test_file.txt'
    var_10 = 'test_file.txt'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.txt'
    var_5 = 'test.txt'
    var_6 = True
    var_7 = 'https://example.com/test.txt'
    var_8 = 'test.txt'
    var_9 = 'https://drive.google.com/file/d/test_id/view'
    var_10 = 'test_id'
    var_11 = 'https://example.com/test.zip'
    var_12 = 'test.zip'
    var_13 = 'https://example.com/test.tar.gz'
    var_14 = 'test.tar.gz'
    var_15 = 'https://example.com/test.unsupported'
    var_16 = 'test.unsupported'



# Parsed testcases at query #10
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/test_id/view'
    var_4 = 'https://example.com/test_file.zip'
    var_5 = None
    var_6 = module_0.download(var_0, var_5, var_1)
    var_7 = module_1.exists(var_6)
    var_8 = 'https://raw.githubusercontent.com/user/repo/main/test_file.txt?raw=true'
    var_9 = module_1.exists(var_6)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'test'
    var_8 = 'https://example.com/test.txt'
    var_9 = 'test.txt'
    var_10 = 'https://drive.google.com/file/d/123456789/view'
    var_11 = '123456789'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = 'test_file.txt'
    var_4 = 'custom_test_file.txt'
    var_5 = None
    var_6 = 'https://drive.google.com/file/d/test_file_id/view'
    var_7 = b'test content'
    var_8 = [var_7]
    var_9 = 'test_file_id'
    var_10 = 'test.tar.gz'
    var_11 = None
    var_12 = True
    var_13 = None
    var_14 = True



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = 'https://example.com/test.zip'
    var_4 = True
    var_5 = 'test'
    var_6 = 'https://drive.google.com/file/d/test_id/view'
    var_7 = 'test_id'
    var_8 = 'https://example.com/nonexistent.txt'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = {}
    var_3 = True
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = 'download_warning_test_id'
    var_7 = 'test_token'
    var_8 = (var_6, var_7)
    var_9 = b'test content'
    var_10 = {}
    var_11 = True



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = 'https://example.com/test.zip'
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/123456789/view'
    var_6 = 'test_gdrive.txt'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = None
    var_3 = 'test.txt'
    var_4 = True
    var_5 = True
    var_6 = 'https://example.com/test.zip'
    var_7 = 'test.zip'
    var_8 = 'https://drive.google.com/file/d/test_id/view'
    var_9 = 'test_id'
    var_10 = 'https://example.com/nonexistent.txt'
    var_11 = 'nonexistent.txt'



# Parsed testcases at query #17
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.zip'
    var_3 = 'test.zip'
    var_4 = True
    var_5 = 'extracted_file'
    var_6 = 'https://example.com/test.txt'
    var_7 = 'test.txt'
    var_8 = 'https://drive.google.com/file/d/test_id/view'
    var_9 = 'test_id'
    var_10 = 'https://example.com/test.txt'
    var_11 = 'test.txt'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'
    var_14 = 'https://example.com/test.txt'
    var_15 = module_0.download(var_14)
    var_16 = module_1.exists(var_15)
    var_17 = 'https://example.com/test.txt'
    var_18 = module_1.exists(var_15)
    var_19 = 'test.txt'



# Parsed testcases at query #18
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = 'https://drive.google.com/file/d/123456789/view'
    var_6 = 'test.txt'
    var_7 = module_0.download(var_0)
    var_8 = module_1.exists(var_7)
    var_9 = module_2.dirname(var_7)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = None
    var_3 = 'https://drive.google.com/file/d/test_id/view'
    var_4 = b'test content'
    var_5 = 'https://example.com/test_file.zip'
    var_6 = 'test_file.zip'
    var_7 = None
    var_8 = True
    var_9 = 'https://example.com/test_file.tar.gz'
    var_10 = 'test_file.tar.gz'
    var_11 = None
    var_12 = True
    var_13 = 'r'
    var_14 = None
    var_15 = True
    var_16 = None
    var_17 = True
    var_18 = 'existing_file.txt'
    var_19 = 'test content'
    var_20 = 'existing_file.txt'



# Parsed testcases at query #20
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://drive.google.com/file/d/test_file_id/view'
    var_3 = 'test_file_id'
    var_4 = 'https://example.com/test_file.txt'
    var_5 = 'test_file.txt'
    var_6 = True
    var_7 = 'https://example.com/test_file.zip'
    var_8 = 'test_file.zip'
    var_9 = 'https://example.com/test_file.tar.gz'
    var_10 = 'test_file.tar.gz'
    var_11 = 'https://example.com/test_file.txt'
    var_12 = 'test_file.txt'
    var_13 = 'https://example.com/test_file.txt'
    var_14 = 'test_file.txt'
    var_15 = 'https://example.com/test_file.txt'
    var_16 = module_0.download(var_15)
    var_17 = module_1.exists(var_16)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/123456789/view'
    var_4 = 'test_gdrive.txt'
    var_5 = 'https://example.com/test.zip'
    var_6 = 'test.zip'
    var_7 = 'extracted_file'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.zip'
    var_3 = 'test.zip'
    var_4 = True
    var_5 = 'extracted_file'
    var_6 = 'https://example.com/test.txt'
    var_7 = 'test_progress.txt'
    var_8 = 'https://drive.google.com/file/d/test_id/view'
    var_9 = 'test_gdrive.txt'
    var_10 = 'test_id'
    var_11 = 'https://example.com/test_default.txt'
    var_12 = 'test_default.txt'
    var_13 = 'https://example.com/test_existing.txt'
    var_14 = 'test_existing.txt'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.txt'
    var_5 = 'test.txt'
    var_6 = True
    var_7 = 'https://example.com/test.zip'
    var_8 = 'test.zip'
    var_9 = 'https://drive.google.com/file/d/123456789/view'
    var_10 = '123456789'
    var_11 = 'https://example.com/test.txt'
    var_12 = 'test.txt'



# Parsed testcases at query #24
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = 'https://drive.google.com/file/d/123456789/view'
    var_6 = 'test_gdrive.txt'
    var_7 = module_0.download(var_3)
    var_8 = module_1.exists(var_7)
    var_9 = module_1.exists(var_7)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.txt'
    var_3 = 'test.txt'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'extracted_file'
    var_8 = 'https://drive.google.com/file/d/test_file_id/view'
    var_9 = 'test_file_id'
    var_10 = 'https://example.com/test.txt'
    var_11 = 'test.txt'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'



# Parsed testcases at query #26
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://drive.google.com/file/d/123456789/view'
    var_3 = '123456789'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'test'
    var_8 = 'https://example.com/test.tar.gz'
    var_9 = 'test.tar.gz'
    var_10 = 'https://example.com/test.txt'
    var_11 = 'test.txt'
    var_12 = 'https://example.com/test.txt'
    var_13 = 'test.txt'
    var_14 = 'https://example.com/test.txt'
    var_15 = 'test.txt'
    var_16 = 'https://example.com/test.txt'
    var_17 = module_0.download(var_16)
    var_18 = module_1.exists(var_17)



# Parsed testcases at query #27
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/123456789/view'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'extracted_file'
    var_7 = 'https://example.com/test.tar.gz'
    var_8 = 'test.tar.gz'
    var_9 = module_0.download(var_0, filename=var_1)
    var_10 = module_1.exists(var_9)
    var_11 = module_2.basename(var_9)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'custom_name.txt'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/test_id/view'
    var_5 = b'test content'
    var_6 = 'test_id'
    var_7 = 'https://example.com/test_file.zip'
    var_8 = True
    var_9 = 'existing.txt'
    var_10 = 'existing content'
    var_11 = 'existing.txt'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://example.com/test_file.zip'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/test_file_id/view'
    var_5 = 'https://example.com/non_existent_file.txt'



