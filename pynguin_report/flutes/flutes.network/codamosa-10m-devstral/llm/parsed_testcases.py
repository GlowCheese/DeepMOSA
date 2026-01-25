####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = {}
    var_3 = 'test_file.txt'
    var_4 = 'custom_name.txt'
    var_5 = {}
    var_6 = 'https://drive.google.com/file/d/test_id/view'
    var_7 = b'test content'
    var_8 = 'test_id'
    var_9 = 'test.zip'
    var_10 = {}
    var_11 = True
    var_12 = {}
    var_13 = True



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = {}
    var_3 = True
    var_4 = True
    var_5 = 'https://drive.google.com/file/d/test_id/view'
    var_6 = b'test content'
    var_7 = [var_6]
    var_8 = True
    var_9 = True



# Parsed testcases at query #3
#--------------------------


import flutes.network as module_0
import genericpath as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://drive.google.com/file/d/123456789/view'
    var_3 = 'google_drive_file.txt'
    var_4 = 'https://example.com/large_file.zip'
    var_5 = 'large_file.zip'
    var_6 = True
    var_7 = 'https://example.com/test_archive.zip'
    var_8 = 'test_archive.zip'
    var_9 = 'extracted_file.txt'
    var_10 = 'https://example.com/test_archive.tar.gz'
    var_11 = 'test_archive.tar.gz'
    var_12 = 'https://example.com/custom_bar_file.txt'
    var_13 = 'custom_bar_file.txt'
    var_14 = 'https://example.com/no_save_dir_file.txt'
    var_15 = 'no_save_dir_file.txt'
    var_16 = module_0.download(var_14, filename=var_15)
    var_17 = module_1.exists(var_16)
    var_18 = module_2.basename(var_16)
    assert var_18 == 'no_save_dir_file.txt'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
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
    var_9 = 'https://drive.google.com/file/d/test_id/view'
    var_10 = 'test_id'
    var_11 = 'https://example.com/test.txt'
    var_12 = 'test.txt'



# Parsed testcases at query #6
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'custom_test.txt'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/123456789/view'
    var_5 = '123456789'
    var_6 = 'https://example.com/test.zip'
    var_7 = 'https://example.com/test.tar.gz'
    var_8 = None
    var_9 = module_0.download(var_0, var_8)
    var_10 = module_1.exists(var_9)
    var_11 = module_1.exists(var_9)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = 'https://drive.google.com/file/d/123456789/view'
    var_6 = 'test_gdrive.txt'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'https://drive.google.com/file/d/test_id/view'



# Parsed testcases at query #9
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
    var_8 = 'test.tar.gz'
    var_9 = None
    var_10 = 'test.tar.gz'
    var_11 = True
    var_12 = 'existing.txt'
    var_13 = 'test'
    var_14 = 'existing.txt'



# Parsed testcases at query #10
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/testfile.txt'
    var_1 = 'testfile.txt'
    var_2 = 'testfile.txt'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/testfileid/view'
    var_5 = 'https://example.com/testfile.zip'
    var_6 = 'extracted_file'
    var_7 = module_0.download(var_0)
    var_8 = module_1.exists(var_7)
    var_9 = module_1.exists(var_7)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = None
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = True
    var_7 = 'https://drive.google.com/file/d/test_id/view'
    var_8 = 'test_file'
    var_9 = 'test'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/test_file_id/view'
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'test_file.zip'
    var_6 = 'extracted_file.txt'
    var_7 = 'https://example.com/test_file.tar.gz'
    var_8 = 'test_file.tar.gz'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/123456789/view'
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'test_file.txt'
    var_3 = True
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'test_file.zip'
    var_6 = 'extracted_file.txt'
    var_7 = 'https://example.com/test_file.tar.gz'
    var_8 = 'test_file.tar.gz'
    var_9 = 'https://drive.google.com/file/d/test_file_id/view'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = True
    var_3 = 'https://drive.google.com/file/d/123456789/view'
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'test_file.zip'
    var_6 = 'https://example.com/nonexistent_file.txt'
    var_7 = 'nonexistent_file.txt'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'https://example.com/test_file.txt'
    var_3 = 'https://example.com/test_file.txt'
    var_4 = 'test_file.txt'
    var_5 = True
    var_6 = 'https://example.com/test_file.txt'
    var_7 = 'test_file.txt'
    var_8 = 'https://example.com/test_file.zip'
    var_9 = 'test_file.zip'
    var_10 = 'https://drive.google.com/file/d/test_file_id/view'
    var_11 = 'test_file_id'



# Parsed testcases at query #17
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
    var_13 = 'https://example.com/test.txt'
    var_14 = 'test.txt'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'https://example.com/test.zip'
    var_3 = True
    var_4 = 'https://drive.google.com/file/d/test_id/view'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'test_file.txt'
    var_3 = True
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'extracted.txt'
    var_6 = 'https://example.com/test_file.tar.gz'
    var_7 = 'https://drive.google.com/file/d/test_file_id/view'
    var_8 = 'test_file_id'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test_file.txt'
    var_1 = 'test_file.txt'
    var_2 = 'test_file.txt'
    var_3 = True
    var_4 = 'https://example.com/test_file.zip'
    var_5 = 'https://drive.google.com/file/d/123456789/view'
    var_6 = '123456789'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = True
    var_3 = 'https://example.com/test.zip'
    var_4 = 'test.zip'
    var_5 = 'https://drive.google.com/file/d/12345/view'
    var_6 = 'test.txt'
    var_7 = 'https://example.com/nonexistent.txt'
    var_8 = 'nonexistent.txt'



# Parsed testcases at query #22
#--------------------------


import flutes.network as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'extracted.txt'
    var_6 = 'https://drive.google.com/file/d/123456789/view'
    var_7 = module_0.download(var_0)
    var_8 = module_1.exists(var_7)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'https://example.com/test.txt'
    var_1 = 'test.txt'
    var_2 = 'test.txt'
    var_3 = True
    var_4 = 'https://example.com/test.zip'
    var_5 = 'test.zip'
    var_6 = 'test'
    var_7 = 'https://example.com/test.tar.gz'
    var_8 = 'test.tar.gz'
    var_9 = 'https://drive.google.com/file/d/123456789/view'
    var_10 = '123456789'
    var_11 = 'https://example.com/nonexistent.txt'
    var_12 = 'nonexistent.txt'



