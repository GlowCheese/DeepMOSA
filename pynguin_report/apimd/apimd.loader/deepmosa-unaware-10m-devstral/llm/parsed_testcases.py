####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'NonExistent'
    var_5 = 'non_existent_package'
    var_6 = {var_4: var_5}
    var_7 = 'Test1'
    var_8 = 'Test2'
    var_9 = 'test_package1'
    var_10 = 'test_package2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'custom_docs'
    var_13 = False
    var_14 = 2



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '# Test package'
    var_2 = '# Test module'
    var_3 = 'subpkg'
    assert var_3 == 0
    var_4 = '# Test subpackage'
    var_5 = 'test_pkg.module'
    var_6 = 'module'
    var_7 = 'test_pkg.subpkg'
    var_8 = 'non_existent_pkg'
    var_9 = 'test_pkg-stubs'
    var_10 = '# Test stub'
    var_11 = 'test_pkg'
    var_12 = 'test_pkg.module'
    var_13 = 'module'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'valid_package'
    var_1 = '/path/to/valid_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = 'invalid_package'
    var_8 = '/path/to/invalid_package'
    var_9 = True
    var_10 = 1
    var_11 = False
    var_12 = module_0.loader(var_7, var_8, var_9, var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'empty_package'
    var_15 = '/path/to/empty_package'
    var_16 = True
    var_17 = 1
    var_18 = False
    var_19 = module_0.loader(var_14, var_15, var_16, var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = 'package_with_stubs'
    var_22 = '/path/to/package_with_stubs'
    var_23 = True
    var_24 = 1
    var_25 = False
    var_26 = module_0.loader(var_21, var_22, var_23, var_24, var_25)
    var_27 = len(var_26)
    var_28 = 'package_with_extensions'
    var_29 = '/path/to/package_with_extensions'
    var_30 = True
    var_31 = 1
    var_32 = False
    var_33 = module_0.loader(var_28, var_29, var_30, var_31, var_32)
    var_34 = len(var_33)



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestTitle'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'NonExistent'
    var_10 = 'non_existent_module'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = {var_0: var_1}
    var_18 = ''
    var_19 = module_0.gen_api(var_17, var_3, prefix=var_18, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = {}
    var_22 = module_0.gen_api(var_21, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = len(var_22)
    assert var_23 == 0



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'NonExistent'
    var_6 = 'non_existent_package'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = {var_0: var_1}
    var_11 = True
    var_12 = module_0.gen_api(var_10, dry=var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = {var_0: var_1}
    var_15 = 'custom_docs'
    var_16 = False
    var_17 = 2
    var_18 = module_0.gen_api(var_14, prefix=var_15, link=var_16, level=var_17, toc=var_11)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'Test1'
    var_21 = 'Test2'
    var_22 = 'test_package1'
    var_23 = 'test_package2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.gen_api(var_24)
    var_26 = len(var_25)
    assert var_26 == 2



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package docstring."""'
    var_2 = 'test_mod'
    var_3 = '"""Test module docstring."""'
    var_4 = 'test_pkg'
    var_5 = True
    var_6 = False



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = '## Test API\n\n'
    var_12 = {}
    var_13 = module_0.gen_api(var_12, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_5)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'NonExistent'
    var_16 = 'non_existent_package'
    var_17 = {var_15: var_16}
    var_18 = module_0.gen_api(var_17, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_5)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = {var_0: var_1}
    var_21 = False
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = var_22[var_21]
    var_25 = {var_0: var_1}
    var_26 = 3
    var_27 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_26, toc=var_5, dry=var_5)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_27[var_21]
    var_30 = '### Test API\n\n'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_module.py'
    var_2 = '\n"""This is a test module."""\ndef test_function():\n    """A test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg.test_module'
    var_7 = '\n"""This is a test module."""\ndef test_function():\n    """A test function."""\n    pass\n'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""'
    var_2 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass'
    var_3 = True
    var_4 = False
    var_5 = 'non_existent_pkg'



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/to/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = len(var_3)
    var_5 = 'empty_pkg'
    var_6 = '/path/to/empty_pkg'
    var_7 = False
    var_8 = module_0.loader(var_5, var_6, var_7, var_2, var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = 'mixed_pkg'
    var_11 = '/path/to/mixed_pkg'
    var_12 = 2
    var_13 = module_0.loader(var_10, var_11, var_2, var_12, var_2)
    var_14 = len(var_13)
    var_15 = 'ext_pkg'
    var_16 = '/path/to/ext_pkg'
    var_17 = module_0.loader(var_15, var_16, var_7, var_2, var_7)
    var_18 = len(var_17)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'Invalid'
    var_5 = 'non_existent_package'
    var_6 = {var_4: var_5}
    var_7 = 'Test1'
    var_8 = 'Test2'
    var_9 = 'test_package1'
    var_10 = 'test_package2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_0: var_1}
    var_13 = False
    var_14 = 2
    var_15 = {var_0: var_1}
    var_16 = 'test-package-api.md'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '\n"""Test module."""\ndef test_function():\n    """Test function."""\n    pass\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_2 = 'test_package'
    var_3 = '__init__.py'



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'NonExistent'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Test1'
    var_13 = 'Test2'
    var_14 = 'test_package1'
    var_15 = 'test_package2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_0: var_1}
    var_20 = 'custom_docs'
    var_21 = False
    var_22 = 2
    var_23 = module_0.gen_api(var_19, prefix=var_20, link=var_21, level=var_22, toc=var_4, dry=var_4)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = {var_0: var_1}
    var_26 = '/custom/path'
    var_27 = module_0.gen_api(var_25, var_26, dry=var_4)
    var_28 = len(var_27)
    assert var_28 == 1



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'importlib.util.find_spec'
    var_1 = 'importlib.util.spec_from_file_location'
    var_2 = 'importlib.util.module_from_spec'
    var_3 = 'importlib.abc.Loader.exec_module'
    var_4 = 'os.walk'
    var_5 = 'os.path.isfile'
    var_6 = 'os.path.isdir'
    var_7 = 'os.mkdir'
    var_8 = 'os.path._write'
    var_9 = 'os.path._read'
    var_10 = 'TestTitle'
    var_11 = 'test_module'
    var_12 = {var_10: var_11}
    var_13 = 'docs'
    var_14 = '# Test Title API\n\nTest content'
    var_15 = []
    var_16 = 'test_file.py'
    var_17 = 'test_file.pyi'
    var_18 = [var_16, var_17]
    var_19 = '.py'
    var_20 = '.pyi'
    var_21 = (var_19, var_20)
    var_22 = 'compiler.Parser'
    var_23 = True
    var_24 = False



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'NonExistent'
    var_7 = 'non_existent_package'
    var_8 = {var_6: var_7}
    var_9 = module_0.gen_api(var_8, dry=var_3)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = 'Test1'
    var_12 = 'Test2'
    var_13 = 'test_package1'
    var_14 = 'test_package2'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.gen_api(var_15, dry=var_3)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = {var_0: var_1}
    var_19 = 'custom_docs'
    var_20 = 2
    var_21 = module_0.gen_api(var_18, prefix=var_19, level=var_20, dry=var_3)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {var_0: var_1}
    var_24 = False
    var_25 = module_0.gen_api(var_23, link=var_24, toc=var_3, dry=var_3)
    var_26 = len(var_25)
    assert var_26 == 1



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 0
    var_5 = '# Test API'
    var_6 = 'NonExistent'
    var_7 = 'non_existent_package'
    var_8 = {var_6: var_7}
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {var_0: var_1}
    var_15 = False
    var_16 = 'test-package-api.md'
    var_17 = {var_0: var_1}
    var_18 = False
    var_19 = 2
    var_20 = '## Test API'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False
    var_6 = '"""Test package."""\n'
    var_7 = 'test_pkg.module'
    var_8 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_9 = 'test_pkg'
    var_10 = True
    var_11 = False
    var_12 = 'test_pkg'
    var_13 = True
    var_14 = False
    var_15 = 'no module for test_pkg in this platform'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_5 = False
    var_6 = 1
    var_7 = 'nonexistent'
    var_8 = 'nonexistent_ext'
    var_9 = False
    var_10 = 1



# Parsed testcases at query #20
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'Invalid'
    var_12 = 'nonexistent_module'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test_module-api.md'
    var_20 = {var_0: var_1}
    var_21 = 'custom_docs'
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_21, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = module_1.isdir(var_21)
    var_25 = {var_0: var_1}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_26, toc=var_6, dry=var_5)
    var_28 = var_27[var_6]
    var_29 = '## Test API'
    var_30 = {var_0: var_1}
    var_31 = module_0.gen_api(var_30, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_32 = '[toc]'
    var_33 = var_31[var_6]
    var_34 = {var_0: var_1}
    var_35 = module_0.gen_api(var_34, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_36 = len(var_35)
    assert var_36 == 1



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Module docstring."""\ndef func():\n    """Function docstring."""\n    pass\n'
    var_2 = []
    var_3 = 'module.py'
    var_4 = [var_3]
    var_5 = 'test_pkg'
    var_6 = False
    var_7 = 1



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# TestPackage API\n\n'
    var_9 = 'TestPackage1'
    var_10 = 'TestPackage2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_6]
    var_17 = '# TestPackage1 API\n\n'
    var_18 = var_14[var_3]
    var_19 = '# TestPackage2 API\n\n'
    var_20 = {var_0: var_1}
    var_21 = 'custom_docs'
    var_22 = False
    var_23 = 2
    var_24 = module_0.gen_api(var_20, prefix=var_21, link=var_22, level=var_23, toc=var_3, dry=var_3)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = var_24[var_22]
    var_27 = '## TestPackage API\n\n'
    var_28 = 'NonExistentPackage'
    var_29 = 'non_existent_package'
    var_30 = {var_28: var_29}
    var_31 = module_0.gen_api(var_30, dry=var_3)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = {var_0: var_1}
    var_34 = '/custom/path'
    var_35 = module_0.gen_api(var_33, var_34, dry=var_3)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_35[var_22]



# Parsed testcases at query #24
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = True
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_4 = 'test_module'
    var_5 = False
    var_6 = module_0.loader(var_4, var_0, var_3, var_3, var_5)



# Parsed testcases at query #25
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'NonExistent'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Test1'
    var_13 = 'Test2'
    var_14 = 'test_package1'
    var_15 = 'test_package2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_0: var_1}
    var_20 = 'custom_docs'
    var_21 = module_0.gen_api(var_19, prefix=var_20, dry=var_4)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {var_0: var_1}
    var_24 = 2
    var_25 = module_0.gen_api(var_23, prefix=var_3, level=var_24, dry=var_4)
    var_26 = {var_0: var_1}
    var_27 = module_0.gen_api(var_26, prefix=var_3, toc=var_4, dry=var_4)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = {var_0: var_1}
    var_30 = False
    var_31 = module_0.gen_api(var_29, prefix=var_3, link=var_30, dry=var_4)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = '__init__.py'
    var_5 = '"""Test package."""'
    var_6 = True
    var_7 = 'NonExistent'
    var_8 = 'non_existent'
    var_9 = {var_7: var_8}
    var_10 = 'another_package'
    var_11 = '"""Another test package."""'
    var_12 = 'AnotherPackage'
    var_13 = {var_0: var_1, var_12: var_10}
    var_14 = 2



# Parsed testcases at query #27
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = {var_0: var_1}
    var_24 = 2
    var_25 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_24, toc=var_6, dry=var_5)
    var_26 = var_25[var_6]
    var_27 = '## Test API\n\n'
    var_28 = {var_0: var_1}
    var_29 = module_0.gen_api(var_28, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_30 = {var_0: var_1}
    var_31 = module_0.gen_api(var_30, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = 'test_docs/test-package-api.md'
    var_34 = module_1.isfile(var_33)



# Parsed testcases at query #28
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'Invalid'
    var_10 = 'invalid_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = {var_0: var_1}
    var_18 = 'new_test_docs'
    var_19 = module_0.gen_api(var_17, var_3, prefix=var_18, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_20 = len(var_19)
    assert var_20 == 1



# Parsed testcases at query #29
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_pkg'
    var_3 = 'test_path'
    var_4 = '.py'
    var_5 = '.so'
    var_6 = (var_4, var_5)
    var_7 = 'test_path'
    var_8 = []
    var_9 = 'test.py'
    var_10 = 'test.so'
    var_11 = [var_9, var_10]
    var_12 = (var_7, var_8, var_11)
    var_13 = '/'
    var_14 = False
    var_15 = 1
    var_16 = module_0.loader(var_2, var_3, var_14, var_15, var_14)
    var_17 = 'test_path/test.so'
    var_18 = 'test_path/'
    var_19 = '.py'
    var_20 = '.so'
    var_21 = (var_19, var_20)
    var_22 = 'test_path'
    var_23 = []
    var_24 = 'test.py'
    var_25 = 'test.so'
    var_26 = [var_24, var_25]
    var_27 = (var_22, var_23, var_26)
    var_28 = '/'
    var_29 = False
    var_30 = 1
    var_31 = module_0.loader(var_2, var_3, var_29, var_30, var_29)
    var_32 = 'test_path/test.so'
    var_33 = 'test_path/'



# Parsed testcases at query #30
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = var_21[var_4]
    var_24 = '## Test1 API'
    var_25 = var_21[var_6]
    var_26 = '## Test2 API'
    var_27 = {var_0: var_1}
    var_28 = module_0.gen_api(var_27, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_4)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'test-package-api.md'



# Parsed testcases at query #31
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, prefix=var_3, dry=var_6)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, prefix=var_3, dry=var_6)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = {var_0: var_1}
    var_24 = 'custom_docs'
    var_25 = module_0.gen_api(var_23, prefix=var_24, dry=var_6)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = {var_0: var_1}
    var_28 = 3
    var_29 = module_0.gen_api(var_27, prefix=var_3, level=var_28, dry=var_6)
    var_30 = var_29[var_4]
    var_31 = '### Test API\n\n'
    var_32 = {var_0: var_1}
    var_33 = module_0.gen_api(var_32, prefix=var_3, dry=var_4)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'test_docs/test-package-api.md'
    var_36 = module_1.isfile(var_35)
    var_37 = {var_0: var_1}
    var_38 = '/invalid/path'
    var_39 = module_0.gen_api(var_37, prefix=var_38, dry=var_6)
    var_40 = len(var_39)
    assert var_40 == 1



# Parsed testcases at query #32
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_pkg'
    var_2 = 'module.py'
    var_3 = '\n"""This is a test module."""\ndef test_function():\n    """A test function."""\n    pass\n'
    var_4 = 'module.pyi'
    var_5 = '\ndef test_function() -> None: ...\n'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_1, var_0, var_6, var_7, var_6)



# Parsed testcases at query #33
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = True
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_4 = 'test_module'
    var_5 = False
    var_6 = module_0.loader(var_4, var_0, var_5, var_3, var_5)



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = False
    var_4 = 1
    var_5 = 'non_existent_pkg'



# Parsed testcases at query #35
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = 'test_mod.py'
    var_4 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_5 = False
    var_6 = module_0.loader(var_2, var_0, var_4, var_4, var_5)



# Parsed testcases at query #36
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_package'
    var_3 = 'def test_function():\n    pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = module_0.parse(var_2, var_3)
    var_6 = 'test_package.submodule'
    var_7 = 'def test_subfunction():\n    pass'
    var_8 = module_0.parse(var_6, var_7)
    var_9 = module_0.parse(var_2, var_3)
    var_10 = module_0.parse(var_2, var_3)
    var_11 = module_0.parse(var_2, var_3)
    var_12 = 'test_module'
    var_13 = ()
    var_14 = '__doc__'
    var_15 = 'Test module'
    var_16 = {var_14: var_15}
    var_17 = type(var_12, var_13, var_16)



# Parsed testcases at query #37
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Invalid'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Test1'
    var_13 = 'Test2'
    var_14 = 'test_package1'
    var_15 = 'test_package2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_0: var_1}
    var_20 = False
    var_21 = 2
    var_22 = module_0.gen_api(var_19, prefix=var_3, link=var_20, level=var_21, toc=var_4, dry=var_4)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = {var_0: var_1}
    var_25 = None
    var_26 = module_0.gen_api(var_24, var_25, prefix=var_3, dry=var_4)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_5 = True
    var_6 = False
    var_7 = 'nonexistent'



# Parsed testcases at query #39
#--------------------------




# Parsed testcases at query #40
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'Test1'
    var_7 = 'Test2'
    var_8 = 'test_package1'
    var_9 = 'test_package2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = {var_0: var_1}
    var_14 = 'custom_docs'
    var_15 = False
    var_16 = 2
    var_17 = module_0.gen_api(var_13, prefix=var_14, link=var_15, level=var_16, toc=var_3, dry=var_3)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'NonExistent'
    var_20 = 'non_existent_package'
    var_21 = {var_19: var_20}
    var_22 = module_0.gen_api(var_21, dry=var_3)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = {var_0: var_1}
    var_25 = '/custom/path'
    var_26 = module_0.gen_api(var_24, var_25, dry=var_3)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #41
#--------------------------




# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_2 = 'test_package'
    var_3 = '__init__.py'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'TestTitle'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = '# TestTitle API\n\nTest content'
    var_5 = 'os.path.isdir'
    var_6 = False
    var_7 = 'os.mkdir'
    var_8 = 'sys.path.append'
    var_9 = 'your_module._site_path'
    var_10 = 'your_module.loader'
    var_11 = 'Test content'
    var_12 = 'your_module._write'
    var_13 = 'your_module.logger.info'
    var_14 = 'your_module.logger.warning'
    var_15 = True



# Parsed testcases at query #44
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0.gen_api(var_0, var_1)
    var_3 = 'Test'
    var_4 = 'nonexistent_package'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, var_1, dry=var_6)
    var_8 = 'OS'
    var_9 = 'os'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10, var_1, dry=var_6)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = '# OS API'
    var_16 = 'OS'
    var_17 = 'os'
    var_18 = {var_16: var_17}
    var_19 = None
    var_20 = False
    var_21 = 'os-api.md'
    var_22 = module_1.isfile(var_10)
    var_23 = 'Sys'
    var_24 = 'sys'
    var_25 = {var_8: var_21, var_23: var_24}
    var_26 = module_0.gen_api(var_25, var_17, dry=var_6)
    var_27 = len(var_26)
    assert var_27 == 2



# Parsed testcases at query #45
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = 'Test1'
    var_19 = 'Test2'
    var_20 = 'test_package1'
    var_21 = 'test_package2'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = module_0.gen_api(var_22, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = {var_0: var_1}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_26, toc=var_6, dry=var_5)
    var_28 = var_27[var_6]
    var_29 = '## Test API\n\n'



# Parsed testcases at query #46
#--------------------------




# Parsed testcases at query #47
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #48
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'NonExistent'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = {}
    var_13 = module_0.gen_api(var_12, prefix=var_3, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Custom'
    var_16 = {var_15: var_1}
    var_17 = 'custom_docs'
    var_18 = False
    var_19 = 2
    var_20 = module_0.gen_api(var_16, prefix=var_17, link=var_18, level=var_19, toc=var_4, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'WithPWD'
    var_23 = {var_22: var_1}
    var_24 = '/custom/path'
    var_25 = 'pwd_docs'
    var_26 = module_0.gen_api(var_23, var_24, prefix=var_25, dry=var_4)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'NonExistent'
    var_5 = 'non_existent_package'
    var_6 = {var_4: var_5}
    var_7 = 'Test1'
    var_8 = 'Test2'
    var_9 = 'test_package1'
    var_10 = 'test_package2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_0: var_1}
    var_13 = False
    var_14 = 2
    var_15 = {var_0: var_1}
    var_16 = None



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package init."""'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function."""\n    pass'
    var_5 = 'test_pkg'
    var_6 = True
    var_7 = False
    var_8 = '"""Test package init."""'
    var_9 = 'test_pkg.module'
    var_10 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function."""\n    pass'
    var_11 = 'test_pkg <= /test_pkg/__init__.py'
    var_12 = 'test_pkg.module <= /test_pkg/module.py'
    var_13 = 'non_existent_pkg'
    var_14 = True
    var_15 = False
    var_16 = 'no module for non_existent_pkg in this platform'



# Parsed testcases at query #51
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API'
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'NonExistent'
    var_17 = 'non_existent_package'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = {var_0: var_1}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_3, dry=var_3)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API'
    var_29 = {}
    var_30 = module_0.gen_api(var_29, dry=var_3)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = {var_0: var_1}
    var_33 = None
    var_34 = module_0.gen_api(var_32, var_33, dry=var_3)
    var_35 = len(var_34)
    assert var_35 == 1



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'Test'
    var_2 = 'non_existent_package'
    var_3 = {var_1: var_2}
    var_4 = 'OS'
    var_5 = 'os'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = 0
    var_9 = '# OS API'
    var_10 = 'Sys'
    var_11 = 'sys'
    var_12 = {var_4: var_5, var_10: var_11}
    var_13 = '#'
    var_14 = 'new_dir'
    var_15 = {var_4: var_5}
    var_16 = {var_4: var_5}
    var_17 = False
    var_18 = 2
    var_19 = '## OS API'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_5 = False
    var_6 = 1



# Parsed testcases at query #54
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = module_1.isdir(var_4)
    var_19 = {var_0: var_1}
    var_20 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_21 = module_1.isdir(var_4)
    var_22 = 'test-package-api.md'
    var_23 = {}
    var_24 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = {var_0: var_1}
    var_27 = None
    var_28 = module_0.gen_api(var_26, var_27, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_29 = len(var_28)
    assert var_29 == 1



# Parsed testcases at query #55
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = True
    var_2 = '"""Module 1 docstring."""\n\ndef func1():\n    """Function 1 docstring."""\n    pass\n'
    var_3 = 'def func2() -> None: ...\n'
    var_4 = 'test_package'
    var_5 = False
    var_6 = module_0.loader(var_4, var_0, var_3, var_3, var_5)



# Parsed testcases at query #56
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'compiled_doc'
    var_6 = 'test_module'
    var_7 = 'path/to/test_module'
    var_8 = (var_6, var_7)
    var_9 = 'test_submodule'
    var_10 = 'path/to/test_submodule'
    var_11 = (var_9, var_10)
    var_12 = 'module_docstring'
    var_13 = 'submodule_docstring'
    var_14 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_14 == 'compiled_doc'
    var_15 = 'non_existent_package'
    var_16 = module_0.loader(var_15, var_1, var_2, var_3, var_4)
    assert var_16 == ''



# Parsed testcases at query #57
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API'
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = {var_0: var_1}
    var_17 = 'test_docs'
    var_18 = 2
    var_19 = module_0.gen_api(var_16, prefix=var_17, level=var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = var_19[var_6]
    var_22 = '## Test API'
    var_23 = 'NonExistent'
    var_24 = 'non_existent_package'
    var_25 = {var_23: var_24}
    var_26 = module_0.gen_api(var_25, dry=var_3)
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = {}
    var_29 = module_0.gen_api(var_28, dry=var_3)
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = {var_0: var_1}
    var_32 = '/custom/path'
    var_33 = module_0.gen_api(var_31, var_32, dry=var_3)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = {var_0: var_1}
    var_36 = False
    var_37 = 3
    var_38 = module_0.gen_api(var_35, prefix=var_17, link=var_36, level=var_37, toc=var_3, dry=var_3)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = var_38[var_36]
    var_41 = '### Test API'



# Parsed testcases at query #58
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'Invalid'
    var_12 = 'invalid_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test_docs/test-package-api.md'
    var_20 = module_1.isfile(var_19)
    var_21 = {var_0: var_1}
    var_22 = None
    var_23 = module_0.gen_api(var_21, var_22, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'Test1'
    var_26 = 'Test2'
    var_27 = 'test_package1'
    var_28 = 'test_package2'
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = module_0.gen_api(var_29, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_31 = len(var_30)
    assert var_31 == 2



# Parsed testcases at query #59
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = '__init__.py'
    var_4 = '"""Test package init."""\n'
    var_5 = 'module.py'
    var_6 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_7 = False
    var_8 = module_0.loader(var_2, var_0, var_7, var_6, var_7)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_5 = True
    var_6 = False
    var_7 = 'nonexistent'



# Parsed testcases at query #61
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test/site-packages'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_module1'
    var_6 = '/path/to/test_module1'
    var_7 = (var_5, var_6)
    var_8 = 'test_module2'
    var_9 = '/path/to/test_module2'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_12 == 'compiled_output'
    var_13 = len(var_11)
    var_14 = 2
    var_15 = var_13 * var_14
    var_16 = len(var_11)
    var_17 = 3
    var_18 = var_16 * var_17



# Parsed testcases at query #62
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = {var_0: var_1}
    var_20 = 2
    var_21 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_20, toc=var_6, dry=var_5)
    var_22 = var_21[var_6]
    var_23 = '## Test API'
    var_24 = {var_0: var_1}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = {var_0: var_1}
    var_28 = module_0.gen_api(var_27, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_29 = {}
    var_30 = module_0.gen_api(var_29, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_31 = len(var_30)
    assert var_31 == 0



# Parsed testcases at query #63
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_pwd'
    var_5 = 'docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_2, var_4, prefix=var_5, link=var_6, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'test_package'
    var_11 = 'test_path'
    var_12 = 'test_pwd'
    var_13 = 'docs'
    var_14 = True
    var_15 = False
    var_16 = module_0.gen_api(var_2, var_12, prefix=var_13, link=var_14, level=var_14, toc=var_15, dry=var_14)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 'test_package'
    var_19 = '='
    var_20 = 12
    var_21 = var_19 * var_20
    var_22 = 'compiled doc'
    var_23 = 'test_pwd'
    var_24 = 'docs'
    var_25 = True
    var_26 = False
    var_27 = module_0.gen_api(var_2, var_23, prefix=var_24, link=var_25, level=var_25, toc=var_26, dry=var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = "'test_package' can not be found"
    var_30 = 'test_path'
    var_31 = 'test_pwd'
    var_32 = 'docs'
    var_33 = True
    var_34 = False
    var_35 = module_0.gen_api(var_2, var_31, prefix=var_32, link=var_33, level=var_33, toc=var_34, dry=var_34)
    var_36 = len(var_35)
    assert var_36 == 1



# Parsed testcases at query #64
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'test_path'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_2, var_3, var_4, var_5, var_4)
    var_7 = 'test_path/test.py'
    var_8 = 'r'
    var_9 = 'test'
    var_10 = 'test_path'
    var_11 = False
    var_12 = 1
    var_13 = module_0.loader(var_9, var_10, var_11, var_12, var_11)



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = '__init__.py'
    var_5 = '"""Test package."""'
    var_6 = True
    var_7 = 0
    var_8 = '# Test API'
    var_9 = False
    var_10 = 'test-package-api.md'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = 'another_package'
    var_15 = '"""Another test package."""'
    var_16 = 'Another'
    var_17 = {var_0: var_1, var_16: var_14}



# Parsed testcases at query #66
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_package'
    var_3 = 'test_path'
    var_4 = '.py'
    var_5 = '.so'
    var_6 = []
    var_7 = 'test_module.py'
    var_8 = 'test_module.so'
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_6, var_9)
    var_11 = '/'
    var_12 = False
    var_13 = 1
    var_14 = module_0.loader(var_2, var_3, var_12, var_13, var_12)
    assert var_14 == 'compiled_doc'
    var_15 = []
    var_16 = 'test_module.txt'
    var_17 = [var_16]
    var_18 = (var_3, var_15, var_17)
    var_19 = False
    var_20 = 1
    var_21 = module_0.loader(var_2, var_3, var_19, var_20, var_19)
    assert var_21 == 'compiled_doc'
    var_22 = '.so'
    var_23 = []
    var_24 = 'test_module.so'
    var_25 = [var_24]
    var_26 = (var_3, var_23, var_25)
    var_27 = '/'
    var_28 = False
    var_29 = 1
    var_30 = module_0.loader(var_2, var_3, var_28, var_29, var_28)
    assert var_30 == 'compiled_doc'



# Parsed testcases at query #67
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_module1'
    var_6 = '/path/to/test_module1'
    var_7 = (var_5, var_6)
    var_8 = 'test_module2'
    var_9 = '/path/to/test_module2'
    var_10 = (var_8, var_9)
    var_11 = 'module1_content'
    var_12 = 'module2_content'
    var_13 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_13 == 'compiled_output'
    var_14 = '/path/to/test_module1.py'
    var_15 = '/path/to/test_module2.py'
    var_16 = (var_5, var_6)
    var_17 = (var_8, var_9)
    var_18 = False
    var_19 = True
    var_20 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_20 == 'compiled_output'
    var_21 = '/path/to/test_module1.so'
    var_22 = '/path/to/test_module2.so'
    var_23 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_23 == 'compiled_output'



# Parsed testcases at query #68
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = '__init__.py'
    var_4 = '"""Test package init."""\n'
    var_5 = 'module.py'
    var_6 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_7 = False
    var_8 = module_0.loader(var_2, var_0, var_7, var_6, var_7)



# Parsed testcases at query #69
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test_docs/test-package-api.md'
    var_20 = module_1.isfile(var_19)
    var_21 = 'Test1'
    var_22 = 'Test2'
    var_23 = 'test_package1'
    var_24 = 'test_package2'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_27 = len(var_26)
    assert var_27 == 2



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_3 = True
    var_4 = False



# Parsed testcases at query #71
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_gen_api_dir'
    var_1 = 'TestPackage'
    var_2 = 'test_package'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, var_0, prefix=var_0, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# TestPackage API'
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, var_0, prefix=var_0, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'TestPackage1'
    var_16 = 'TestPackage2'
    var_17 = 'test_package1'
    var_18 = 'test_package2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.gen_api(var_19, var_0, prefix=var_0, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = {var_1: var_2}
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_22, var_0, prefix=var_0, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## TestPackage API'



# Parsed testcases at query #72
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'Invalid'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = 'test-package-api.md'
    var_19 = {var_0: var_1}
    var_20 = 2
    var_21 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_20, toc=var_6, dry=var_5)
    var_22 = var_21[var_6]
    var_23 = '## Test API'
    var_24 = {var_0: var_1}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_26 = '[toc]'
    var_27 = var_25[var_6]
    var_28 = {var_0: var_1}
    var_29 = module_0.gen_api(var_28, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_30 = len(var_29)
    assert var_30 == 1



# Parsed testcases at query #73
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_package.module1'
    var_6 = '/path/to/test/test_package/module1'
    var_7 = (var_5, var_6)
    var_8 = 'test_package.module2'
    var_9 = '/path/to/test/test_package/module2'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = 'def test_function():\n    pass'
    var_13 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_13 == 'compiled_output'
    var_14 = 'test_package.module1'
    var_15 = 'test_package.module2'
    var_16 = 'test_package.module1 <= /path/to/test/test_package/module1.py'
    var_17 = 'test_package.module2 <= /path/to/test/test_package/module2.py'



# Parsed testcases at query #74
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'Invalid'
    var_12 = 'invalid_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = {var_0: var_1}
    var_24 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'test-package-api.md'



# Parsed testcases at query #75
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[var_5]
    var_9 = '# Test API'
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Test1'
    var_16 = 'Test2'
    var_17 = 'test_package1'
    var_18 = 'test_package2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.gen_api(var_19, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = {var_0: var_1}
    var_23 = 'custom_prefix'
    var_24 = module_0.gen_api(var_22, prefix=var_23, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = {var_0: var_1}
    var_27 = 2
    var_28 = module_0.gen_api(var_26, prefix=var_3, link=var_4, level=var_27, toc=var_5, dry=var_4)
    var_29 = var_28[var_5]
    var_30 = '## Test API'
    var_31 = {var_0: var_1}
    var_32 = module_0.gen_api(var_31, prefix=var_3, link=var_4, level=var_4, toc=var_4, dry=var_4)
    var_33 = len(var_32)
    assert var_33 == 1



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True



# Parsed testcases at query #77
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'test_path'
    var_2 = module_0.gen_api(var_0, var_1)
    var_3 = 'Test'
    var_4 = 'non_existent_package'
    var_5 = {var_3: var_4}
    var_6 = module_0.gen_api(var_5, var_1)
    var_7 = 'os'
    var_8 = {var_3: var_7}
    var_9 = True
    var_10 = module_0.gen_api(var_8, var_1, dry=var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = {var_3: var_7}
    var_13 = 'custom_docs'
    var_14 = module_0.gen_api(var_12, var_1, prefix=var_13)
    var_15 = module_1.isdir(var_13)
    var_16 = 'os-api.md'
    var_17 = {var_3: var_7}
    var_18 = 2
    var_19 = module_0.gen_api(var_17, var_1, level=var_18)
    var_20 = {var_3: var_7}
    var_21 = module_0.gen_api(var_20, var_1, toc=var_9)
    var_22 = '[toc]'
    var_23 = 0
    var_24 = var_21[var_23]
    var_25 = {var_3: var_7}
    var_26 = False
    var_27 = module_0.gen_api(var_25, var_1, link=var_26)



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "'''Test package'''\n"
    var_3 = 'module.py'
    var_4 = "'''Test module'''\n\ndef func():\n    '''Test function'''\n    pass\n"
    var_5 = 'Test'
    var_6 = {var_5: var_0}
    var_7 = True
    var_8 = 'test-pkg-api.md'



# Parsed testcases at query #79
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = 'test_docs_dry_false'
    var_18 = module_0.gen_api(var_16, var_3, prefix=var_17, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_19 = module_1.isdir(var_17)
    var_20 = {var_0: var_1}
    var_21 = 'invalid_path/test_docs'
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_21, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = module_1.isdir(var_21)
    var_24 = {}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 0
    var_27 = {var_0: var_1}
    var_28 = None
    var_29 = module_0.gen_api(var_27, var_28, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '"""Test package init"""'
    var_2 = '"""Module 1 docstring"""'
    var_3 = '"""Module 2 stub"""'
    var_4 = True
    var_5 = False



# Parsed testcases at query #81
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pwd'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'compiled_doc'
    var_6 = 'test_pkg.module1'
    var_7 = 'test_pwd/test_pkg/module1'
    var_8 = (var_6, var_7)
    var_9 = 'test_pkg.module2'
    var_10 = 'test_pwd/test_pkg/module2'
    var_11 = (var_9, var_10)
    var_12 = '# Module 1 docstring'
    var_13 = '# Module 1 stub'
    var_14 = '# Module 2 docstring'
    var_15 = '# Module 2 stub'
    var_16 = True
    var_17 = False
    var_18 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_18 == 'compiled_doc'



# Parsed testcases at query #82
#--------------------------




# Parsed testcases at query #83
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API\n\n'
    var_9 = 'NonExistent'
    var_10 = 'nonexistent'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, dry=var_3)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'Test1'
    var_15 = 'Test2'
    var_16 = 'test1'
    var_17 = 'test2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = {var_0: var_1}
    var_22 = 'custom_docs'
    var_23 = module_0.gen_api(var_21, prefix=var_22, dry=var_3)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = {var_0: var_1}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, level=var_26, dry=var_3)
    var_28 = var_27[var_6]
    var_29 = '## Test API\n\n'
    var_30 = {var_0: var_1}
    var_31 = module_0.gen_api(var_30, toc=var_3, dry=var_3)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = {var_0: var_1}
    var_34 = False
    var_35 = module_0.gen_api(var_33, link=var_34, dry=var_3)
    var_36 = len(var_35)
    assert var_36 == 1



# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = '__init__.py'
    var_5 = '"""Test package."""'
    var_6 = 'module.py'
    var_7 = '"""A test module."""\ndef func():\n    """A test function."""\n    pass'
    var_8 = 'sys.path'
    var_9 = 'os.path.dirname'
    var_10 = True
    var_11 = False
    var_12 = 'test-package-api.md'
    var_13 = 'NonExistent'
    var_14 = 'nonexistent_package'
    var_15 = {var_13: var_14}



# Parsed testcases at query #85
#--------------------------




# Parsed testcases at query #86
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test/site-packages'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_module1'
    var_6 = '/path/to/test/site-packages/test_module1'
    var_7 = (var_5, var_6)
    var_8 = 'test_module2'
    var_9 = '/path/to/test/site-packages/test_module2'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_12 == 'compiled_output'
    var_13 = 'test_module1'
    var_14 = 'content_of_/path/to/test/site-packages/test_module1.py'
    var_15 = 'test_module2'
    var_16 = 'content_of_/path/to/test/site-packages/test_module2.py'



# Parsed testcases at query #87
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'def test_func():\n    pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test_package'
    var_6 = module_0.parse(var_5, var_3)
    var_7 = 'test_package.submodule'
    var_8 = 'def test_func2():\n    pass'
    var_9 = module_0.parse(var_7, var_8)
    var_10 = module_0.parse(var_5, var_3)
    var_11 = module_0.parse(var_7, var_8)
    var_12 = 'test_package.subpackage'
    var_13 = 'def test_func3():\n    pass'
    var_14 = module_0.parse(var_12, var_13)
    var_15 = module_0.parse(var_5, var_3)
    var_16 = module_0.parse(var_7, var_8)
    var_17 = module_0.parse(var_12, var_13)
    var_18 = 'test_package.subpackage.submodule'
    var_19 = 'def test_func4():\n    pass'
    var_20 = module_0.parse(var_18, var_19)
    var_21 = module_0.parse(var_5, var_3)
    var_22 = module_0.parse(var_7, var_8)
    var_23 = module_0.parse(var_12, var_13)
    var_24 = module_0.parse(var_18, var_19)
    var_25 = 'test_package.subpackage.subpackage'
    var_26 = 'def test_func5():\n    pass'
    var_27 = module_0.parse(var_25, var_26)



# Parsed testcases at query #88
#--------------------------




# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False



# Parsed testcases at query #90
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'Test1'
    var_7 = 'Test2'
    var_8 = 'test_package1'
    var_9 = 'test_package2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = {var_0: var_1}
    var_14 = 'custom_docs'
    var_15 = False
    var_16 = 2
    var_17 = module_0.gen_api(var_13, prefix=var_14, link=var_15, level=var_16, toc=var_3, dry=var_3)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'NonExistent'
    var_20 = 'non_existent_package'
    var_21 = {var_19: var_20}
    var_22 = module_0.gen_api(var_21, dry=var_3)
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = {var_0: var_1}
    var_25 = '/custom/path'
    var_26 = module_0.gen_api(var_24, var_25, dry=var_3)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #91
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_module'
    var_3 = 'def test_function():\n    pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test_package'
    var_6 = module_0.parse(var_5, var_3)
    var_7 = 'test_package.submodule'
    var_8 = 'def test_sub_function():\n    pass'
    var_9 = module_0.parse(var_7, var_8)
    var_10 = 'test_class'
    var_11 = 'class TestClass:\n    def test_method(self):\n        pass'
    var_12 = module_0.parse(var_10, var_11)
    var_13 = 'test_docstring'
    var_14 = 'def test_function():\n    """This is a test function."""\n    pass'
    var_15 = module_0.parse(var_13, var_14)
    var_16 = 'test_class_docstring'
    var_17 = 'class TestClass:\n    """This is a test class."""\n    def test_method(self):\n        """This is a test method."""\n        pass'
    var_18 = module_0.parse(var_16, var_17)



# Parsed testcases at query #92
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[var_5]
    var_9 = '# Test API'
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = {}
    var_16 = module_0.gen_api(var_15, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = {var_0: var_1}
    var_19 = '/invalid/path'
    var_20 = module_0.gen_api(var_18, prefix=var_19, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #93
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'test_docs'
    var_3 = True
    var_4 = False
    var_5 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_6 = 'Test'
    var_7 = 'nonexistent_package'
    var_8 = {var_6: var_7}
    var_9 = module_0.gen_api(var_8, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_10 = 'OS'
    var_11 = 'os'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = var_13[var_4]
    var_16 = '# OS API\n\n'
    var_17 = 'Sys'
    var_18 = 'sys'
    var_19 = {var_10: var_11, var_17: var_18}
    var_20 = module_0.gen_api(var_19, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = '#'
    var_23 = {var_10: var_11}
    var_24 = 'another_test_docs'
    var_25 = module_0.gen_api(var_23, var_1, prefix=var_24, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_4]
    var_28 = {var_10: var_11}
    var_29 = 2
    var_30 = module_0.gen_api(var_28, var_1, prefix=var_2, link=var_3, level=var_29, toc=var_4, dry=var_3)
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = var_30[var_4]
    var_33 = '## OS API\n\n'
    var_34 = {var_10: var_11}
    var_35 = module_0.gen_api(var_34, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_3, dry=var_3)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_35[var_4]
    var_38 = {var_10: var_11}
    var_39 = module_0.gen_api(var_38, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_4)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = 'os-api.md'



# Parsed testcases at query #94
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = module_0.gen_api(var_1, dry=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'nonexistent'
    var_6 = {var_5: var_5}
    var_7 = module_0.gen_api(var_6, dry=var_2)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = 'test1'
    var_10 = 'test2'
    var_11 = {var_9: var_9, var_10: var_10}
    var_12 = module_0.gen_api(var_11, dry=var_2)
    var_13 = len(var_12)
    var_14 = {var_0: var_0}
    var_15 = 'custom_docs'
    var_16 = module_0.gen_api(var_14, prefix=var_15, dry=var_2)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = {var_0: var_0}
    var_19 = False
    var_20 = 2
    var_21 = module_0.gen_api(var_18, link=var_19, level=var_20, toc=var_2, dry=var_2)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {}
    var_24 = module_0.gen_api(var_23, dry=var_2)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = {var_0: var_0}
    var_27 = None
    var_28 = module_0.gen_api(var_26, var_27, dry=var_2)
    var_29 = len(var_28)
    assert var_29 == 1



# Parsed testcases at query #95
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pwd'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'test_module'
    var_7 = 'Simple docstring'



# Parsed testcases at query #96
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = {}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'NonExistent'
    var_15 = 'non_existent_package'
    var_16 = {var_14: var_15}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = {var_0: var_1}
    var_20 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'test-package-api.md'



# Parsed testcases at query #97
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_3 = 'test_module.pyi'
    var_4 = '\ndef test_function() -> None: ...\n'
    var_5 = 'test_module'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_5, var_0, var_6, var_6, var_7)
    var_9 = 'non_existent_module'
    var_10 = module_0.loader(var_9, var_0, var_6, var_6, var_7)



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""'
    var_2 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass'
    var_3 = False
    var_4 = 1



# Parsed testcases at query #99
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = '"""Test package init."""\n'
    var_4 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_5 = False
    var_6 = module_0.loader(var_2, var_0, var_5, var_4, var_5)



# Parsed testcases at query #100
#--------------------------


def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'
    var_2 = '"""Test module."""'
    var_3 = 'submodule.py'
    var_4 = '"""Submodule."""\ndef func():\n    """A function."""\n    pass'
    var_5 = 'Test'
    var_6 = {var_5: var_0}
    var_7 = True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '# Test package'
    var_3 = 'module.py'
    var_4 = 'def test_func(): pass'
    var_5 = 'Test'
    var_6 = {var_5: var_0}
    var_7 = 'docs'
    var_8 = True
    var_9 = {var_5: var_0}
    var_10 = False
    var_11 = 'test-pkg-api.md'



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_path'
    var_2 = 'test_docs'
    var_3 = module_0.gen_api(var_0, var_1, prefix=var_2)
    var_4 = 'Test'
    var_5 = 'non_existent_package'
    var_6 = {var_4: var_5}
    var_7 = module_0.gen_api(var_6, var_1, prefix=var_2)
    var_8 = 'Mocked API content'
    var_9 = 'mock_site_path'
    var_10 = False
    var_11 = 'Test'
    var_12 = 'test_package'
    var_13 = {var_11: var_12}
    var_14 = 'test_path'
    var_15 = 'test_docs'
    var_16 = True
    var_17 = module_0.gen_api(var_13, var_14, prefix=var_15, dry=var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'mock_site_path'
    var_20 = False
    var_21 = 'Test'
    var_22 = 'test_package'
    var_23 = {var_21: var_22}
    var_24 = 'test_path'
    var_25 = 'test_docs'
    var_26 = False
    var_27 = module_0.gen_api(var_23, var_24, prefix=var_25, dry=var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'mocked_path'
    var_30 = '# Test API\n\nMocked API content'



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_package'
    var_2 = '"""Test package init."""\n'
    var_3 = 'test_module.py'
    var_4 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_5 = False
    var_6 = 1
    var_7 = module_0.loader(var_1, var_0, var_5, var_6, var_5)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '# Test package'
    var_3 = 'module1.py'
    var_4 = '# Module 1'
    var_5 = 'subpkg'
    var_6 = '# Subpackage'
    var_7 = 'module2.py'
    var_8 = '# Module 2'
    var_9 = 'module3.pyi'
    var_10 = '# Stub file'
    var_11 = 'test_pkg.subpkg'
    var_12 = 'test_pkg.subpkg.module2'
    var_13 = 'module2'
    var_14 = 'test_pkg.subpkg.module3'
    var_15 = 'module3'
    var_16 = 'stub_pkg-stubs'
    var_17 = 'module4.pyi'
    var_18 = '# Stub module'
    var_19 = 'stub_pkg'
    var_20 = 'stub_pkg.module4'
    var_21 = 'module4'
    var_22 = 'invalid_pkg'
    var_23 = 'not_python.txt'
    var_24 = '# Not Python'
    var_25 = 'module5.pyc'
    var_26 = '# Bytecode'
    var_27 = 'empty_pkg'



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package docstring."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_1, var_0, var_6, var_7, var_6)



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'Invalid'
    var_10 = 'nonexistent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = {var_0: var_1}
    var_18 = 'new_test_docs'
    var_19 = module_0.gen_api(var_17, var_3, prefix=var_18, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_20 = len(var_19)
    assert var_20 == 1



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Invalid'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Test1'
    var_13 = 'Test2'
    var_14 = 'test_package1'
    var_15 = 'test_package2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = {var_0: var_1}
    var_20 = False
    var_21 = 2
    var_22 = module_0.gen_api(var_19, prefix=var_3, link=var_20, level=var_21, toc=var_4, dry=var_4)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = {var_0: var_1}
    var_25 = '/some/path'
    var_26 = module_0.gen_api(var_24, var_25, prefix=var_3, dry=var_4)
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'non_existent_package'
    var_7 = 'test_path'
    var_8 = True
    var_9 = 1
    var_10 = False
    var_11 = module_0.loader(var_6, var_7, var_8, var_9, var_10)
    var_12 = 'test_package'
    var_13 = 'test_path'
    var_14 = False
    var_15 = 2
    var_16 = True
    var_17 = module_0.loader(var_12, var_13, var_14, var_15, var_16)



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    var_9 = {}
    var_10 = module_0.gen_api(var_9, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'NonExistent'
    var_13 = 'non_existent_package'
    var_14 = {var_12: var_13}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = {var_0: var_1}
    var_18 = module_0.gen_api(var_17, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = {}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'NonExistent'
    var_15 = 'non_existent_package'
    var_16 = {var_14: var_15}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = {var_0: var_1}
    var_20 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = module_1.isdir(var_4)
    var_23 = {var_0: var_1}
    var_24 = 2
    var_25 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_24, toc=var_6, dry=var_5)
    var_26 = var_25[var_6]
    var_27 = '## Test API'



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_pkg'
    var_3 = 'test_path'
    var_4 = 'test_content'
    var_5 = '.py'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_2, var_3, var_6, var_7, var_6)
    assert var_8 == 'compiled_doc'



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_package'
    var_3 = 'test_path'
    var_4 = 'test_path'
    var_5 = []
    var_6 = 'test.py'
    var_7 = 'test.pyi'
    var_8 = 'test.so'
    var_9 = [var_6, var_7, var_8]
    var_10 = (var_4, var_5, var_9)
    var_11 = '.py'
    var_12 = '.pyi'
    var_13 = '.so'
    var_14 = (var_11, var_12, var_13)
    var_15 = [var_4]
    var_16 = True
    var_17 = False
    var_18 = module_0.loader(var_2, var_3, var_16, var_16, var_17)
    var_19 = True
    var_20 = False
    var_21 = module_0.loader(var_2, var_3, var_19, var_19, var_20)
    assert var_21 == ''



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_package'
    var_3 = '"""Test package init."""\n'
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_6 = False
    var_7 = module_0.loader(var_2, var_0, var_5, var_5, var_6)



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_package'
    var_3 = 'test_package_path'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'test error'
    var_7 = 'test_package'
    var_8 = 'test_path'
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_7, var_8, var_9, var_10, var_9)
    var_12 = 'test_package'
    var_13 = 'test_path'
    var_14 = False
    var_15 = 1
    var_16 = module_0.loader(var_12, var_13, var_14, var_15, var_14)



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_pwd'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'Invalid'
    var_10 = 'nonexistent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'test-package-api.md'
    var_18 = {var_0: var_1}
    var_19 = 2
    var_20 = module_0.gen_api(var_18, var_3, prefix=var_4, link=var_6, level=var_19, toc=var_5, dry=var_5)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_pkg'
    var_3 = 'test_path'
    var_4 = '.py'
    var_5 = 'test_path'
    var_6 = []
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = (var_5, var_6, var_8)
    var_10 = '/'
    var_11 = True
    var_12 = False
    var_13 = module_0.loader(var_2, var_3, var_11, var_11, var_12)
    var_14 = '.so'
    var_15 = 'test_path'
    var_16 = []
    var_17 = 'test.so'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = '/'
    var_21 = True
    var_22 = False
    var_23 = module_0.loader(var_2, var_3, var_21, var_21, var_22)
    var_24 = 'test_path'
    var_25 = []
    var_26 = 'test.py'
    var_27 = [var_26]
    var_28 = (var_24, var_25, var_27)
    var_29 = '/'
    var_30 = True
    var_31 = False
    var_32 = module_0.loader(var_2, var_3, var_30, var_30, var_31)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '# Test package'
    var_3 = 'module.py'
    var_4 = '"""Module docstring"""'
    var_5 = 'sys.path'
    var_6 = 'Test'
    var_7 = {var_6: var_0}
    var_8 = True
    var_9 = {var_6: var_0}
    var_10 = 'test-pkg-api.md'
    var_11 = 'Missing'
    var_12 = 'nonexistent'
    var_13 = {var_11: var_12}



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'NonExistent'
    var_9 = 'non_existent_package'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = 'Test1'
    var_14 = 'Test2'
    var_15 = 'test_package1'
    var_16 = 'test_package2'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.gen_api(var_17, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_19 = len(var_18)
    var_20 = {var_0: var_1}
    var_21 = 2
    var_22 = module_0.gen_api(var_20, prefix=var_3, link=var_4, level=var_21, toc=var_5, dry=var_4)
    var_23 = {var_0: var_1}
    var_24 = module_0.gen_api(var_23, prefix=var_3, link=var_4, level=var_4, toc=var_4, dry=var_4)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = {var_0: var_1}
    var_27 = module_0.gen_api(var_26, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_5)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'test_docs/test-package-api.md'
    var_30 = module_1.isfile(var_29)
    var_31 = {var_0: var_1}
    var_32 = 'custom_path'
    var_33 = module_0.gen_api(var_31, var_32, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_34 = len(var_33)



# Parsed testcases at query #19
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# Test API'
    var_10 = 'Invalid'
    var_11 = 'invalid_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, prefix=var_3, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Test1'
    var_16 = 'Test2'
    var_17 = 'test_package1'
    var_18 = 'test_package2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.gen_api(var_19, prefix=var_3, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = {var_0: var_1}
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_22, prefix=var_3, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API'
    var_29 = {var_0: var_1}
    var_30 = 'non_existent_docs'
    var_31 = module_0.gen_api(var_29, prefix=var_30, dry=var_4)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = {var_0: var_1}
    var_34 = '/some/path'
    var_35 = module_0.gen_api(var_33, var_34, dry=var_4)
    var_36 = len(var_35)
    assert var_36 == 1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '__init__.py'
    var_4 = '"""Test package."""'
    var_5 = True
    var_6 = False
    var_7 = 'test-package-api.md'
    var_8 = 'NonExistent'
    var_9 = 'non_existent_package'
    var_10 = {var_8: var_9}
    var_11 = 'test_package2'
    var_12 = '"""Second test package."""'
    var_13 = 'Test2'
    var_14 = {var_0: var_1, var_13: var_11}



# Parsed testcases at query #21
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'NonExistent'
    var_10 = 'non_existent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = module_0.gen_api(var_14, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'Test1'
    var_18 = 'Test2'
    var_19 = 'test_package1'
    var_20 = 'test_package2'
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.gen_api(var_21, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #22
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = module_1.isdir(var_4)
    var_20 = 'test_docs/test-package-api.md'
    var_21 = module_1.isfile(var_20)
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = {var_0: var_1}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_26, toc=var_6, dry=var_5)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_27[var_6]
    var_30 = '## Test API'
    var_31 = {var_0: var_1}
    var_32 = module_0.gen_api(var_31, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_33 = len(var_32)
    assert var_33 == 1



# Parsed testcases at query #23
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[var_5]
    var_9 = '# Test API\n\n'
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Test1'
    var_16 = 'Test2'
    var_17 = 'test_package1'
    var_18 = 'test_package2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.gen_api(var_19, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_5)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'test-package-api.md'
    var_26 = {var_0: var_1}
    var_27 = 2
    var_28 = module_0.gen_api(var_26, prefix=var_3, link=var_4, level=var_27, toc=var_5, dry=var_4)
    var_29 = var_28[var_5]
    var_30 = '## Test API\n\n'
    var_31 = {var_0: var_1}
    var_32 = module_0.gen_api(var_31, prefix=var_3, link=var_4, level=var_4, toc=var_4, dry=var_4)
    var_33 = {var_0: var_1}
    var_34 = module_0.gen_api(var_33, prefix=var_3, link=var_5, level=var_4, toc=var_5, dry=var_4)
    var_35 = len(var_34)
    assert var_35 == 1



# Parsed testcases at query #24
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test_title'
    var_1 = 'test_name'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = '# test_title API\n\n'
    var_8 = {var_0: var_1}
    var_9 = 'custom_docs'
    var_10 = module_0.gen_api(var_8, prefix=var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = module_1.isdir(var_9)
    var_13 = {var_0: var_1}
    var_14 = True
    var_15 = module_0.gen_api(var_13, dry=var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'non_existent'
    var_18 = 'non_existent_package'
    var_19 = {var_17: var_18}
    var_20 = module_0.gen_api(var_19)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = 'title1'
    var_23 = 'title2'
    var_24 = 'pkg1'
    var_25 = 'pkg2'
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.gen_api(var_26)
    var_28 = len(var_27)
    var_29 = 'test'
    var_30 = {var_29: var_29}
    var_31 = False
    var_32 = 2
    var_33 = module_0.gen_api(var_30, link=var_31, level=var_32, toc=var_14)
    var_34 = len(var_33)



# Parsed testcases at query #25
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'NonExistent'
    var_10 = 'non_existent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'Test1'
    var_15 = 'Test2'
    var_16 = 'test_package1'
    var_17 = 'test_package2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = {var_0: var_1}
    var_22 = module_0.gen_api(var_21, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = {var_0: var_1}
    var_25 = 'invalid_path'
    var_26 = module_0.gen_api(var_24, var_3, prefix=var_25, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = {var_0: var_1}
    var_29 = 2
    var_30 = module_0.gen_api(var_28, var_3, prefix=var_4, link=var_5, level=var_29, toc=var_6, dry=var_5)
    var_31 = len(var_30)
    assert var_31 == 1



# Parsed testcases at query #26
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = 'test_package'
    var_2 = '__init__.py'
    var_3 = "'''Test package'''\n"
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_1, var_0, var_6, var_7, var_6)



# Parsed testcases at query #27
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test-package-api.md'
    var_20 = 'Test1'
    var_21 = 'Test2'
    var_22 = 'test_package1'
    var_23 = 'test_package2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.gen_api(var_24, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = {var_0: var_1}
    var_28 = '/custom/path'
    var_29 = module_0.gen_api(var_27, var_28, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_30 = len(var_29)
    assert var_30 == 1



# Parsed testcases at query #28
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_3 = 'test_module'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_3, var_0, var_4, var_5, var_4)



# Parsed testcases at query #29
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API'
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_6]
    var_17 = '# Test1 API'
    var_18 = var_14[var_3]
    var_19 = '# Test2 API'
    var_20 = 'NonExistent'
    var_21 = 'non_existent_package'
    var_22 = {var_20: var_21}
    var_23 = module_0.gen_api(var_22, dry=var_3)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = {var_0: var_1}
    var_26 = 'test_docs'
    var_27 = False
    var_28 = 2
    var_29 = module_0.gen_api(var_25, prefix=var_26, link=var_27, level=var_28, toc=var_3, dry=var_3)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = var_29[var_27]
    var_32 = '## Test API'
    var_33 = {var_0: var_1}
    var_34 = '/some/path'
    var_35 = module_0.gen_api(var_33, var_34, dry=var_3)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_35[var_27]



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = "\n        '''Test module docstring.'''\n        def test_function():\n            '''Test function docstring.'''\n            pass\n    "
    var_2 = 'TestModule'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = 1
    var_7 = True
    var_8 = 'NonExistent'
    var_9 = 'non_existent'
    var_10 = {var_8: var_9}
    var_11 = True
    var_12 = {var_2: var_3}
    var_13 = 'test-module-api.md'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False



# Parsed testcases at query #33
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_package'
    var_3 = 'test_path'
    var_4 = module_0.loader(var_2, var_3, var_0, var_0, var_1)
    var_5 = 'non_existent_package'
    var_6 = 'test_path'
    var_7 = module_0.loader(var_5, var_6, var_0, var_0, var_1)
    var_8 = 'test_extension'
    var_9 = 'test_path'
    var_10 = module_0.loader(var_8, var_9, var_0, var_0, var_1)



# Parsed testcases at query #34
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_3 = 'test_module'
    var_4 = True
    var_5 = False
    var_6 = module_0.loader(var_3, var_0, var_4, var_4, var_5)



# Parsed testcases at query #35
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = {var_0: var_1}
    var_24 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = module_1.isdir(var_4)
    var_27 = {var_0: var_1}
    var_28 = 2
    var_29 = module_0.gen_api(var_27, var_3, prefix=var_4, link=var_5, level=var_28, toc=var_6, dry=var_5)
    var_30 = var_29[var_6]
    var_31 = '## Test API'
    var_32 = {var_0: var_1}
    var_33 = module_0.gen_api(var_32, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_34 = len(var_33)
    assert var_34 == 1



# Parsed testcases at query #36
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_module1'
    var_6 = 'test_path1'
    var_7 = (var_5, var_6)
    var_8 = 'test_module2'
    var_9 = 'test_path2'
    var_10 = (var_8, var_9)
    var_11 = 'module1_content'
    var_12 = 'module2_content'
    var_13 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_13 == 'compiled_output'



# Parsed testcases at query #37
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Module 1."""\ndef func1():\n    """Function 1."""\n    pass\n'
    var_3 = 'def func2() -> None: ...\n'
    var_4 = 'test_pkg'
    var_5 = True
    var_6 = False
    var_7 = 'non_existent_pkg'
    var_8 = '/fake/path'
    var_9 = False
    var_10 = 2
    var_11 = True
    var_12 = module_0.loader(var_7, var_8, var_9, var_10, var_11)
    assert var_12 == ''



# Parsed testcases at query #38
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test Module'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Non Existent'
    var_8 = 'non_existent_module'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Module 1'
    var_13 = 'Module 2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'Custom'
    var_20 = 'custom_module'
    var_21 = {var_19: var_20}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'module.py'
    var_2 = '\n"""This is a test module."""\ndef test_function():\n    """This is a test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg.module'
    var_7 = '"""This is a test module."""\ndef test_function():\n    """This is a test function."""\n    pass'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_3 = 'test_package'
    var_4 = True
    var_5 = False



# Parsed testcases at query #41
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = False
    var_4 = 1
    var_5 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_6 = 'non_existent_module'
    var_7 = '/non/existent/path'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)
    assert var_10 == ''
    assert var_10 == 'Compiled extension output'
    var_11 = 'test_extension.cpython-38-x86_64-linux-gnu.so'
    var_12 = ''
    var_13 = 'test_extension'
    var_14 = False
    var_15 = 1



# Parsed testcases at query #42
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# Test API'
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, prefix=var_3, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Test1'
    var_16 = 'Test2'
    var_17 = 'test_package1'
    var_18 = 'test_package2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.gen_api(var_19, prefix=var_3, dry=var_4)
    var_21 = len(var_20)
    var_22 = {var_0: var_1}
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_22, prefix=var_3, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API'
    var_29 = {var_0: var_1}
    var_30 = None
    var_31 = module_0.gen_api(var_29, var_30, prefix=var_3, dry=var_4)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #43
#--------------------------




# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'TestTitle'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = '# TestTitle API\n\nTest documentation'
    var_5 = 'os.path.isdir'
    var_6 = False
    var_7 = 'os.mkdir'
    var_8 = 'os.path.join'
    var_9 = 'test_module-api.md'
    var_10 = 'sys.path.append'
    var_11 = 'compiler.loader'
    var_12 = 'compiler._write'
    var_13 = 'compiler._site_path'
    var_14 = True



# Parsed testcases at query #45
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test_docs/test-package-api.md'
    var_20 = module_1.isfile(var_19)
    var_21 = 'Test1'
    var_22 = 'Test2'
    var_23 = 'test_package1'
    var_24 = 'test_package2'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = {var_0: var_1}
    var_29 = 'custom_docs'
    var_30 = module_0.gen_api(var_28, var_3, prefix=var_29, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_31 = len(var_30)
    assert var_31 == 1



# Parsed testcases at query #46
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_package'
    var_3 = '"""Test package init."""\n'
    var_4 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_5 = False
    var_6 = module_0.loader(var_2, var_0, var_4, var_4, var_5)



# Parsed testcases at query #47
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API'
    var_9 = 'NonExistent'
    var_10 = 'non_existent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, dry=var_3)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'Test1'
    var_15 = 'Test2'
    var_16 = 'test_package1'
    var_17 = 'test_package2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = {var_0: var_1}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_3, dry=var_3)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API'
    var_29 = {var_0: var_1}
    var_30 = '/custom/path'
    var_31 = module_0.gen_api(var_29, var_30, dry=var_3)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #48
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API'
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_6]
    var_17 = '# Test1 API'
    var_18 = var_14[var_3]
    var_19 = '# Test2 API'
    var_20 = {var_0: var_1}
    var_21 = 'custom_docs'
    var_22 = 2
    var_23 = module_0.gen_api(var_20, prefix=var_21, level=var_22, dry=var_3)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_23[var_6]
    var_26 = '## Test API'
    var_27 = 'NonExistent'
    var_28 = 'non_existent_package'
    var_29 = {var_27: var_28}
    var_30 = module_0.gen_api(var_29, dry=var_3)
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = {var_0: var_1}
    var_33 = False
    var_34 = 3
    var_35 = module_0.gen_api(var_32, link=var_33, level=var_34, toc=var_3, dry=var_3)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_35[var_33]
    var_38 = '### Test API'



# Parsed testcases at query #49
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestTitle'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# TestTitle API'
    var_11 = 'InvalidTitle'
    var_12 = 'invalid_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = None
    var_18 = module_0.gen_api(var_16, var_17, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_19 = {var_0: var_1}
    var_20 = 2
    var_21 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_20, toc=var_6, dry=var_5)
    var_22 = var_21[var_6]
    var_23 = '## TestTitle API'
    var_24 = {var_0: var_1}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)



# Parsed testcases at query #50
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass'
    var_3 = 'test_pkg'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_3, var_0, var_4, var_5, var_4)
    var_7 = 'non_existent_pkg'
    var_8 = module_0.loader(var_7, var_0, var_4, var_5, var_4)



# Parsed testcases at query #51
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = 'Invalid'
    var_8 = 'nonexistent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = {}
    var_13 = module_0.gen_api(var_12, prefix=var_3, dry=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = 'Custom'
    var_16 = {var_15: var_1}
    var_17 = False
    var_18 = 2
    var_19 = module_0.gen_api(var_16, prefix=var_3, link=var_17, level=var_18, toc=var_4, dry=var_4)
    var_20 = len(var_19)
    var_21 = var_19[var_17]
    var_22 = '## Custom API\n\n'
    var_23 = 'PWD'
    var_24 = {var_23: var_1}
    var_25 = 'some_path'
    var_26 = module_0.gen_api(var_24, var_25, prefix=var_3, dry=var_4)
    var_27 = len(var_26)



# Parsed testcases at query #52
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'NonExistent'
    var_8 = 'non_existent_module'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'CustomModule'
    var_20 = 'custom_module'
    var_21 = {var_19: var_20}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1



# Parsed testcases at query #53
#--------------------------




# Parsed testcases at query #54
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_path'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = '.pyi'
    var_5 = (var_3, var_4)
    var_6 = 'test_package'
    var_7 = 'test_pwd'
    var_8 = True
    var_9 = module_0.loader(var_6, var_7, var_8, var_8, var_8)
    assert var_9 == 'compiled_output'
    var_10 = 'test_path.py'



# Parsed testcases at query #55
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = 'os'
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)
    var_7 = len(var_6)
    var_8 = {var_0: var_3}
    var_9 = len(var_6)
    var_10 = 'os-api.md'
    var_11 = '# test API'
    var_12 = 'test1'
    var_13 = 'test2'
    var_14 = 'sys'
    var_15 = {var_12: var_3, var_13: var_14}
    var_16 = len(var_6)
    assert var_16 == 2
    var_17 = 'sys-api.md'
    var_18 = {var_0: var_3}
    var_19 = False
    var_20 = 2
    var_21 = len(var_6)



# Parsed testcases at query #56
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = 'Test1'
    var_17 = 'Test2'
    var_18 = 'test_package1'
    var_19 = 'test_package2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = {var_0: var_1}
    var_24 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'test_docs/test-package-api.md'
    var_27 = module_1.isfile(var_26)
    var_28 = {var_0: var_1}
    var_29 = 2
    var_30 = module_0.gen_api(var_28, var_3, prefix=var_4, link=var_5, level=var_29, toc=var_6, dry=var_5)
    var_31 = var_30[var_6]
    var_32 = '## Test API'
    var_33 = {var_0: var_1}
    var_34 = module_0.gen_api(var_33, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_35 = {var_0: var_1}
    var_36 = module_0.gen_api(var_35, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_37 = len(var_36)
    assert var_37 == 1



# Parsed testcases at query #57
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'Invalid'
    var_10 = 'invalid_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'Test1'
    var_15 = 'Test2'
    var_16 = 'test_package1'
    var_17 = 'test_package2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = {var_0: var_1}
    var_22 = module_0.gen_api(var_21, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'test-package-api.md'



# Parsed testcases at query #58
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = '# Test API\n\n'
    var_8 = 'NonExistent'
    var_9 = 'non_existent_package'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = 'Test1'
    var_14 = 'Test2'
    var_15 = 'test_package1'
    var_16 = 'test_package2'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.gen_api(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = {var_0: var_1}
    var_21 = 'custom_docs'
    var_22 = module_0.gen_api(var_20, prefix=var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = {var_0: var_1}
    var_25 = True
    var_26 = module_0.gen_api(var_24, dry=var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = {var_0: var_1}
    var_29 = 2
    var_30 = module_0.gen_api(var_28, level=var_29)
    var_31 = var_30[var_5]
    var_32 = '## Test API\n\n'
    var_33 = {var_0: var_1}
    var_34 = module_0.gen_api(var_33, toc=var_25)
    var_35 = len(var_34)
    assert var_35 == 1



# Parsed testcases at query #59
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'NonExistent'
    var_10 = 'non_existent_package'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, prefix=var_3, dry=var_6)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'Test1'
    var_15 = 'Test2'
    var_16 = 'test_package1'
    var_17 = 'test_package2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, prefix=var_3, dry=var_6)
    var_20 = len(var_19)
    var_21 = {var_0: var_1}
    var_22 = 'custom_docs'
    var_23 = module_0.gen_api(var_21, prefix=var_22, dry=var_6)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = {var_0: var_1}
    var_26 = 3
    var_27 = module_0.gen_api(var_25, prefix=var_3, level=var_26, dry=var_6)
    var_28 = {var_0: var_1}
    var_29 = module_0.gen_api(var_28, prefix=var_3, toc=var_6, dry=var_6)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = {var_0: var_1}
    var_32 = module_0.gen_api(var_31, prefix=var_3, dry=var_6)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = {var_0: var_1}
    var_35 = 'some_path'
    var_36 = module_0.gen_api(var_34, var_35, dry=var_6)
    var_37 = len(var_36)



# Parsed testcases at query #60
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test Package'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Non-existent'
    var_8 = 'non_existent_package'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, prefix=var_3, dry=var_4)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'Package1'
    var_13 = 'Package2'
    var_14 = 'test_package1'
    var_15 = 'test_package2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, prefix=var_3, dry=var_4)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'Custom'
    var_20 = 'test_custom'
    var_21 = {var_19: var_20}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1



# Parsed testcases at query #61
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = 'new_test_docs'
    var_18 = False
    var_19 = 2
    var_20 = True
    var_21 = module_0.gen_api(var_16, prefix=var_17, link=var_18, level=var_19, toc=var_20, dry=var_18)
    var_22 = module_1.isdir(var_17)
    var_23 = {var_18: var_19}
    var_24 = '/custom/path'
    var_25 = module_0.gen_api(var_23, var_24, prefix=var_20, link=var_21, level=var_22, toc=var_6, dry=var_6)



# Parsed testcases at query #62
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'compiled_output'
    var_6 = 'test_module'
    var_7 = 'module_content'
    var_8 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_8 == 'compiled_output'
    var_9 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_9 == 'compiled_output'
    var_10 = 'test_module'
    var_11 = 'module_content'



# Parsed testcases at query #63
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/test'
    var_1 = 'test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'Compiled test string'
    var_6 = module_0.loader(var_1, var_0, var_2, var_3, var_4)
    assert var_6 == 'Compiled test string'



# Parsed testcases at query #64
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_site_packages'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #65
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_loader_dir'
    var_1 = 'test_package'
    var_2 = '"""Test package init."""'
    var_3 = 'test_module.py'
    var_4 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_1, var_0, var_5, var_5, var_6)



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = False
    var_5 = 1
    var_6 = 'non_existent_pkg'
    var_7 = False
    var_8 = 1
    var_9 = 'test_pkg'
    var_10 = '"""Test package."""\n'
    var_11 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_12 = 'test_pkg'
    var_13 = False
    var_14 = 1



# Parsed testcases at query #67
#--------------------------




# Parsed testcases at query #68
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/fake/pwd'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'test_package'
    var_11 = '/fake/path'
    var_12 = 'test_docs/test-package-api.md'
    var_13 = '## Test API\n\ncompiled_doc'
    var_14 = {var_3: var_4}
    var_15 = '/fake/pwd'
    var_16 = 'test_docs'
    var_17 = True
    var_18 = 2
    var_19 = False
    var_20 = module_0.gen_api(var_14, var_15, prefix=var_16, link=var_17, level=var_18, toc=var_17, dry=var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'test_package'
    var_23 = '/fake/path'
    var_24 = 'test_docs/test-package-api.md'
    var_25 = '## Test API\n\ncompiled_doc'
    var_26 = {var_15: var_16}
    var_27 = '/fake/pwd'
    var_28 = 'test_docs'
    var_29 = True
    var_30 = 2
    var_31 = module_0.gen_api(var_26, var_27, prefix=var_28, link=var_29, level=var_30, toc=var_29, dry=var_29)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = 'test_package'
    var_34 = '/fake/path'
    var_35 = '='
    var_36 = 12
    var_37 = var_35 * var_36
    var_38 = '## Test API\n\ncompiled_doc'
    var_39 = {var_27: var_28}



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '# Test package\n"""Test module docstring"""'
    var_3 = 'submodule.py'
    var_4 = 'def test_func(): pass\n"""Submodule docstring"""'
    var_5 = 'sys.path'
    var_6 = 'sys.path.append'
    var_7 = None
    var_8 = 'os.path.isdir'
    var_9 = True
    var_10 = 'os.mkdir'
    var_11 = 'os.path.join'
    var_12 = lambda *args: str(tmp_path.joinpath(*args))
    var_13 = 'os.path.isfile'
    var_14 = '.py'
    var_15 = lambda x: str(tmp_path.joinpath(x)).endswith(var_14)
    var_16 = 'builtins.open'
    var_17 = 'importlib.util.find_spec'
    var_18 = 'importlib.util.spec_from_file_location'
    var_19 = lambda x: var_7
    var_20 = 'importlib.abc.Loader'
    var_21 = 'Test'
    var_22 = {var_21: var_0}
    var_23 = False



# Parsed testcases at query #70
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = {var_0: var_1}
    var_20 = None
    var_21 = module_0.gen_api(var_19, var_20, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)



# Parsed testcases at query #71
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = 'test_docs_dry_false'
    var_18 = module_0.gen_api(var_16, var_3, prefix=var_17, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = {var_0: var_1}
    var_21 = 'invalid_prefix'
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_21, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = {}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 0



# Parsed testcases at query #72
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'Test1'
    var_7 = 'Test2'
    var_8 = 'test_package1'
    var_9 = 'test_package2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 'Invalid'
    var_14 = 'nonexistent_package'
    var_15 = {var_13: var_14}
    var_16 = module_0.gen_api(var_15, dry=var_3)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = {var_0: var_1}
    var_19 = 'custom_docs'
    var_20 = 2
    var_21 = module_0.gen_api(var_18, prefix=var_19, level=var_20, dry=var_3)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {var_0: var_1}
    var_24 = module_0.gen_api(var_23, toc=var_3, dry=var_3)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = {var_0: var_1}
    var_27 = False
    var_28 = module_0.gen_api(var_26, link=var_27, dry=var_3)
    var_29 = len(var_28)
    assert var_29 == 1



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'NonExistent'
    var_5 = 'non_existent_package'
    var_6 = {var_4: var_5}
    var_7 = 'Test1'
    var_8 = 'Test2'
    var_9 = 'test_package1'
    var_10 = 'test_package2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'new_dir'
    var_13 = False
    var_14 = 2
    var_15 = '##'



# Parsed testcases at query #74
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = 'Test'
    var_3 = 'nonexistent_package'
    var_4 = {var_2: var_3}
    var_5 = module_0.gen_api(var_4)
    var_6 = 'os'
    var_7 = {var_2: var_6}
    var_8 = True
    var_9 = module_0.gen_api(var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = {var_2: var_6}
    var_12 = 'test_docs'
    var_13 = module_0.gen_api(var_11, prefix=var_12, dry=var_8)
    var_14 = module_1.isdir(var_12)
    var_15 = {var_2: var_6}
    var_16 = 2
    var_17 = module_0.gen_api(var_15, level=var_16)
    var_18 = 0
    var_19 = var_17[var_18]
    var_20 = '##'



# Parsed testcases at query #75
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API\n\n'
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'NonExistent'
    var_17 = 'non_existent_package'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = {var_0: var_1}
    var_22 = 'custom_docs'
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_21, prefix=var_22, link=var_23, level=var_24, toc=var_3, dry=var_3)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API\n\n'
    var_29 = {var_0: var_1}
    var_30 = '/custom/path'
    var_31 = module_0.gen_api(var_29, var_30, dry=var_3)
    var_32 = len(var_31)
    assert var_32 == 1



# Parsed testcases at query #76
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test-package-api.md'
    var_20 = 'Test1'
    var_21 = 'Test2'
    var_22 = 'test_package1'
    var_23 = 'test_package2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = {var_0: var_1}
    var_28 = 2
    var_29 = module_0.gen_api(var_27, var_3, prefix=var_4, link=var_5, level=var_28, toc=var_6, dry=var_5)
    var_30 = var_29[var_6]
    var_31 = '## Test API\n\n'
    var_32 = {var_0: var_1}
    var_33 = module_0.gen_api(var_32, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_34 = {var_0: var_1}
    var_35 = module_0.gen_api(var_34, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = {var_0: var_1}
    var_38 = None
    var_39 = module_0.gen_api(var_37, var_38, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_40 = len(var_39)
    assert var_40 == 1



# Parsed testcases at query #77
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'Test1'
    var_12 = 'Test2'
    var_13 = 'test_package1'
    var_14 = 'test_package2'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.gen_api(var_15, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'Invalid'
    var_19 = 'nonexistent_package'
    var_20 = {var_18: var_19}
    var_21 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = {var_0: var_1}
    var_24 = 2
    var_25 = module_0.gen_api(var_23, var_3, prefix=var_4, link=var_6, level=var_24, toc=var_5, dry=var_5)
    var_26 = var_25[var_6]
    var_27 = '## Test API'
    var_28 = {var_0: var_1}
    var_29 = module_0.gen_api(var_28, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_30 = module_1.isdir(var_4)



# Parsed testcases at query #78
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = 'NonExistent'
    var_7 = 'non_existent_module'
    var_8 = {var_6: var_7}
    var_9 = 'Extension'
    var_10 = 'test_extension'
    var_11 = {var_9: var_10}
    var_12 = True
    var_13 = module_0.gen_api(var_11, var_2, prefix=var_3, dry=var_12)



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 0
    var_5 = '# Test API\n\n'
    var_6 = 'NonExistent'
    var_7 = 'non_existent_package'
    var_8 = {var_6: var_7}
    var_9 = 'Test1'
    var_10 = 'Test2'
    var_11 = 'test_package1'
    var_12 = 'test_package2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = {var_0: var_1}
    var_15 = 'custom'
    var_16 = {var_0: var_1}
    var_17 = 2
    var_18 = '## Test API\n\n'
    var_19 = {var_0: var_1}
    var_20 = {var_0: var_1}
    var_21 = False



# Parsed testcases at query #80
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API'
    var_11 = 'NonExistent'
    var_12 = 'non_existent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test-package-api.md'
    var_20 = 'Test1'
    var_21 = 'Test2'
    var_22 = 'test_package1'
    var_23 = 'test_package2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = var_25[var_6]
    var_28 = '# Test1 API'
    var_29 = var_25[var_5]
    var_30 = '# Test2 API'
    var_31 = {var_0: var_1}
    var_32 = 2
    var_33 = module_0.gen_api(var_31, var_3, prefix=var_4, link=var_5, level=var_32, toc=var_6, dry=var_5)
    var_34 = var_33[var_6]
    var_35 = '## Test API'



# Parsed testcases at query #81
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'NonExistent'
    var_6 = 'non_existent_package'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = {var_0: var_1}
    var_11 = True
    var_12 = module_0.gen_api(var_10, dry=var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = {var_0: var_1}
    var_15 = 'custom_docs'
    var_16 = module_0.gen_api(var_14, prefix=var_15)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = {var_0: var_1}
    var_19 = False
    var_20 = 2
    var_21 = module_0.gen_api(var_18, link=var_19, level=var_20, toc=var_11)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = {var_0: var_1}
    var_24 = '/custom/path'
    var_25 = module_0.gen_api(var_23, var_24)
    var_26 = len(var_25)
    assert var_26 == 1



# Parsed testcases at query #82
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_6 = False
    var_7 = module_0.loader(var_2, var_0, var_5, var_5, var_6)



# Parsed testcases at query #83
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_package'
    var_3 = 'test_package'
    var_4 = 'test_package'
    var_5 = []
    var_6 = '__init__.py'
    var_7 = [var_6]
    var_8 = (var_4, var_5, var_7)
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_2, var_3, var_9, var_10, var_9)
    var_12 = 'test_extension'
    var_13 = 'test_extension'
    var_14 = 'test_extension'
    var_15 = []
    var_16 = '__init__.pyi'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = False
    var_20 = 1
    var_21 = module_0.loader(var_12, var_13, var_19, var_20, var_19)
    var_22 = 'nonexistent'
    var_23 = 'nonexistent'
    var_24 = False
    var_25 = 1
    var_26 = module_0.loader(var_22, var_23, var_24, var_25, var_24)
    assert var_26 == ''



# Parsed testcases at query #84
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'compiled_doc'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_6 == 'compiled_doc'
    var_7 = 'test_module'
    var_8 = 'module_content'
    var_9 = 'test_extension'
    var_10 = 'path/to/test_extension'
    var_11 = False
    var_12 = 2
    var_13 = True
    var_14 = module_0.loader(var_9, var_10, var_11, var_12, var_13)
    assert var_14 == 'compiled_doc'
    var_15 = 'non_existent'
    var_16 = 'path/to/non_existent'
    var_17 = True
    var_18 = 1
    var_19 = False
    var_20 = ''
    var_21 = module_0.loader(var_15, var_16, var_17, var_18, var_19)
    assert var_21 == ''



