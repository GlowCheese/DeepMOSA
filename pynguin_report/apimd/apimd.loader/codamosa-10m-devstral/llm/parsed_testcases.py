####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    def test_method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = 'test_module'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_3, var_0, var_4, var_5, var_4)



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_5 = []
    var_6 = 'non_existent_module'
    var_7 = '/fake/path'
    var_8 = True
    var_9 = module_0.loader(var_6, var_7, var_8, var_8, var_8)
    assert var_9 == ''
    assert var_9 == 'Compiled output'
    var_10 = 'no module for non_existent_module in this platform'
    var_11 = 'test_extension.so'
    var_12 = 'test_extension'
    var_13 = True



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'testpkg'
    var_1 = 'module.py'
    var_2 = '"""This is a test module."""\n\ndef test_function():\n    """This is a test function."""\n    pass\n'
    var_3 = False
    var_4 = 1
    var_5 = 'nonexistent'
    var_6 = 'emptypkg'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_module.py'
    var_2 = '\n"""\nThis is a test module.\n"""\ndef test_function():\n    """This is a test function."""\n    pass\n'
    var_3 = True
    var_4 = False
    var_5 = 'non_existent_pkg'



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = True
    var_12 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '__init__.py'
    var_4 = '"""Test package."""'
    var_5 = True
    var_6 = 'NonExistent'
    var_7 = 'non_existent'
    var_8 = {var_6: var_7}
    var_9 = 'Another'
    var_10 = 'another'
    var_11 = {var_0: var_1, var_9: var_10}
    var_12 = '"""Another package."""'
    var_13 = 'TestPackage API'
    var_14 = 'Another API'



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_packages'
    var_1 = True
    var_2 = 'test_pkg'
    var_3 = '"""Test package."""\n'
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_6 = False
    var_7 = module_0.loader(var_2, var_0, var_5, var_5, var_6)



# Parsed testcases at query #8
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
    assert var_21 == 2
    var_22 = {var_0: var_1}
    var_23 = False
    var_24 = 2
    var_25 = module_0.gen_api(var_22, prefix=var_3, link=var_23, level=var_24, toc=var_4, dry=var_4)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_25[var_23]
    var_28 = '## Test API'
    var_29 = {}
    var_30 = module_0.gen_api(var_29, prefix=var_3, dry=var_4)
    var_31 = len(var_30)
    assert var_31 == 0



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = {var_2: var_3}
    var_7 = False
    var_8 = 'test-module-api.md'
    var_9 = 'NonExistent'
    var_10 = 'nonexistent_module'
    var_11 = {var_9: var_10}
    var_12 = 'test_module2.py'
    var_13 = '"""Second test module."""\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_14 = 'First'
    var_15 = 'Second'
    var_16 = 'test_module2'
    var_17 = {var_14: var_3, var_15: var_16}
    var_18 = 'test-module2-api.md'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'compiler.walk_packages'
    var_1 = 'test_module'
    var_2 = 'test_path'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.sub'
    var_5 = 'test_path_sub'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 'compiler._site_path'
    var_9 = 'test_site_path'
    var_10 = 'compiler._load_module'
    var_11 = True
    var_12 = 'compiler._read'
    var_13 = 'test docstring'
    var_14 = 'compiler._write'
    var_15 = 'compiler.Parser.new'
    var_16 = 'Test'
    var_17 = {var_16: var_1}
    var_18 = 'test_pwd'
    var_19 = False
    var_20 = 2
    var_21 = {var_16: var_1}
    var_22 = {var_16: var_1}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\ndef func():\n    """Test function."""\n    pass'
    var_5 = 'Test'
    var_6 = True
    var_7 = 'NonExistent'
    var_8 = 'non_existent_pkg'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 'test-pkg-api.md'
    var_12 = 'test_pkg2'
    var_13 = '"""Second test package."""'
    var_14 = 'Test1'
    var_15 = 'Test2'



# Parsed testcases at query #12
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
    var_9 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_9 == 'compiled_doc'
    var_10 = 'path/to/test_module.py'
    var_11 = 'test_module'
    var_12 = 'module_content'
    var_13 = 'test_extension'
    var_14 = 'path/to/test_extension'
    var_15 = (var_13, var_14)
    var_16 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_16 == 'compiled_doc'
    var_17 = 'path/to/test_extension.py'
    var_18 = 'test_extension'
    var_19 = 'path/to/test_extension.so'
    var_20 = 'non_existent_module'
    var_21 = 'path/to/non_existent_module'
    var_22 = (var_20, var_21)
    var_23 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_23 == 'compiled_doc'
    var_24 = 'path/to/non_existent_module.py'
    var_25 = 'non_existent_module'
    var_26 = 'path/to/non_existent_module.so'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""'
    var_2 = '"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg <= '
    var_7 = 'test_pkg.module <= '
    var_8 = 'non_existent_pkg'
    var_9 = True
    var_10 = False
    var_11 = "'non_existent_pkg' can not be found"
    var_12 = 'ext_pkg'
    var_13 = '"""Test extension package."""'
    var_14 = 'ext_module.cpython-38-x86_64-linux-gnu.so'
    var_15 = ''
    var_16 = 'ext_pkg'
    var_17 = True
    var_18 = False
    var_19 = 'loading extension module for fully documented:'



# Parsed testcases at query #14
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'def test_func():\n    pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test_module.submodule'
    var_6 = 'class TestClass:\n    pass\n'
    var_7 = module_0.parse(var_5, var_6)



# Parsed testcases at query #15
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
    var_20 = {var_0: var_1}
    var_21 = 'custom_docs'
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_21, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = var_22[var_6]
    var_25 = {var_0: var_1}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_26, toc=var_6, dry=var_5)
    var_28 = var_27[var_6]
    var_29 = '## Test API'
    var_30 = {var_0: var_1}
    var_31 = module_0.gen_api(var_30, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = {var_0: var_1}
    var_34 = module_0.gen_api(var_33, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = var_34[var_6]



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_package_path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'test_module'
    var_6 = 'test_module_path'
    var_7 = (var_5, var_6)
    var_8 = 'test_submodule'
    var_9 = 'test_submodule_path'
    var_10 = (var_8, var_9)
    var_11 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_11 == 'compiled_output'
    var_12 = 'test_module_path.py'
    var_13 = 'test_module_path.pyi'
    var_14 = 'test_submodule_path.py'
    var_15 = 'test_submodule_path.pyi'
    var_16 = 'test_module'
    var_17 = 'test_module_path.so'
    var_18 = 'test_submodule'
    var_19 = 'test_submodule_path.so'
    var_20 = 'no module for test_module in this platform'



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

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
    var_11 = 'Test1'
    var_12 = 'Test2'
    var_13 = 'test_package1'
    var_14 = 'test_package2'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.gen_api(var_15, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_16[var_6]
    var_19 = '# Test1 API\n\n'
    var_20 = var_16[var_5]
    var_21 = '# Test2 API\n\n'
    var_22 = 'NonExistent'
    var_23 = 'non_existent_package'
    var_24 = {var_22: var_23}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_26 = len(var_25)
    assert var_26 == 0
    var_27 = {var_0: var_1}
    var_28 = module_0.gen_api(var_27, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'test-package-api.md'
    var_31 = {var_0: var_1}
    var_32 = 2
    var_33 = module_0.gen_api(var_31, var_3, prefix=var_4, link=var_5, level=var_32, toc=var_6, dry=var_5)
    var_34 = var_33[var_6]
    var_35 = '## Test API\n\n'
    var_36 = {var_0: var_1}
    var_37 = module_0.gen_api(var_36, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_38 = {var_0: var_1}
    var_39 = module_0.gen_api(var_38, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_40 = len(var_39)
    assert var_40 == 1



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Test package."""\n'
    var_2 = '"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_pkg'
    var_4 = False
    var_5 = 1



# Parsed testcases at query #5
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
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = {var_0: var_1}
    var_20 = 2
    var_21 = module_0.gen_api(var_19, var_3, prefix=var_4, link=var_5, level=var_20, toc=var_6, dry=var_5)
    var_22 = var_21[var_6]
    var_23 = '## Test API\n\n'
    var_24 = {var_0: var_1}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_26 = {var_0: var_1}
    var_27 = module_0.gen_api(var_26, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_28 = len(var_27)
    assert var_28 == 1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'os.path.isdir'
    var_1 = False
    var_2 = 'os.mkdir'
    var_3 = 'os.path.join'
    var_4 = 'test-api.md'
    var_5 = 'sys.path.append'
    var_6 = 'os.path.isfile'
    var_7 = True
    var_8 = 'builtins.open'
    var_9 = 'test content'
    var_10 = 'importlib.util.find_spec'
    var_11 = 'os.path.dirname'
    var_12 = 'os.walk'
    var_13 = []
    var_14 = 'test.py'
    var_15 = [var_14]
    var_16 = 'os.path.abspath'
    var_17 = 'os.sep'
    var_18 = '/'
    var_19 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_20 = '.so'
    var_21 = [var_20]
    var_22 = 'importlib.util.spec_from_file_location'
    var_23 = None
    var_24 = 'sys.path'
    var_25 = []
    var_26 = 'parser.Parser'
    var_27 = 'Test'
    var_28 = 'test'
    var_29 = {var_27: var_28}



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""Module docstring."""\n\ndef func():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test_pkg'
    var_3 = True
    var_4 = False



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    var_10 = 'NonExistent'
    var_11 = 'non_existent_package'
    var_12 = {var_10: var_11}
    var_13 = module_0.gen_api(var_12, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = {}
    var_16 = module_0.gen_api(var_15, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = {var_0: var_1}
    var_19 = None
    var_20 = module_0.gen_api(var_18, var_19, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'test_dir'
    var_2 = module_0.gen_api(var_0, var_1)
    var_3 = 'Test'
    var_4 = 'non_existent_package'
    var_5 = {var_3: var_4}
    var_6 = module_0.gen_api(var_5, var_1)
    var_7 = 'OS'
    var_8 = 'os'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = module_0.gen_api(var_9, var_1, dry=var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = '# OS API'
    var_16 = 'OS'
    var_17 = 'os'
    var_18 = {var_16: var_17}
    var_19 = False
    var_20 = 'os-api.md'
    var_21 = module_1.isfile(var_9)
    var_22 = 'Sys'
    var_23 = 'sys'
    var_24 = {var_7: var_20, var_22: var_23}
    var_25 = module_0.gen_api(var_24, var_17, dry=var_21)
    var_26 = len(var_25)
    assert var_26 == 2



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_dir'
    var_2 = '# Test API\n\n## Module: test_pkg\n\n### Function: test_func\n\nTest function.'
    var_3 = 'test_dir'
    var_4 = []
    var_5 = 'test_pkg.py'
    var_6 = 'test_pkg.pyi'
    var_7 = 'test_pkg.cpython-38-x86_64-linux-gnu.so'
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_3, var_4, var_8)
    var_10 = True
    var_11 = False
    var_12 = module_0.loader(var_0, var_1, var_10, var_10, var_11)
    var_13 = True
    var_14 = False



# Parsed testcases at query #11
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
    var_16 = 'non_existent_package'
    var_17 = {var_0: var_16}
    var_18 = module_0.gen_api(var_17, dry=var_3)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = {var_0: var_1}
    var_21 = 'test_docs'
    var_22 = False
    var_23 = 2
    var_24 = module_0.gen_api(var_20, prefix=var_21, link=var_22, level=var_23, toc=var_3, dry=var_3)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = var_24[var_22]
    var_27 = '## Test API'
    var_28 = {var_0: var_1}
    var_29 = '/some/path'
    var_30 = module_0.gen_api(var_28, var_29, dry=var_3)
    var_31 = len(var_30)
    assert var_31 == 1



# Parsed testcases at query #12
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
    var_12 = 'nonexistent_package'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'test-package-api.md'
    var_20 = {var_0: var_1}
    var_21 = 2
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_21, toc=var_6, dry=var_5)
    var_23 = var_22[var_6]
    var_24 = '## Test API'
    var_25 = {var_0: var_1}
    var_26 = module_0.gen_api(var_25, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_27 = {var_0: var_1}
    var_28 = module_0.gen_api(var_27, var_3, prefix=var_4, link=var_6, level=var_5, toc=var_6, dry=var_5)
    var_29 = len(var_28)
    assert var_29 == 1



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'compiled_output'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_6 == 'compiled_output'
    var_7 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_7 == 'compiled_output'
    var_8 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_8 == 'compiled_output'



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'Compiled result'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_6 == 'Compiled result'



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_pkg'
    var_3 = 'test_path'
    var_4 = True
    var_5 = False
    var_6 = module_0.loader(var_2, var_3, var_4, var_4, var_5)
    var_7 = 'Test docstring'
    var_8 = 'test_pkg'
    var_9 = 'test_path'
    var_10 = True
    var_11 = False
    var_12 = module_0.loader(var_8, var_9, var_10, var_10, var_11)
    var_13 = '.py'
    var_14 = var_9 + var_13
    var_15 = 'test_pkg'
    var_16 = 'test_path'
    var_17 = True
    var_18 = False
    var_19 = module_0.loader(var_15, var_16, var_17, var_17, var_18)
    var_20 = '.py'
    var_21 = var_16 + var_20



# Parsed testcases at query #17
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
    var_6 = 'First'
    var_7 = 'Second'
    var_8 = 'first_package'
    var_9 = 'second_package'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 'Custom'
    var_14 = 'custom_package'
    var_15 = {var_13: var_14}
    var_16 = 'test_docs'
    var_17 = False
    var_18 = 2
    var_19 = module_0.gen_api(var_15, prefix=var_16, link=var_17, level=var_18, toc=var_3, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'NonExistent'
    var_22 = 'non_existent_package'
    var_23 = {var_21: var_22}
    var_24 = module_0.gen_api(var_23, dry=var_3)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = {}
    var_27 = module_0.gen_api(var_26, dry=var_3)
    var_28 = len(var_27)
    assert var_28 == 0



