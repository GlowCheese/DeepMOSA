####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'pkg_path'
    var_2 = 'root'
    var_3 = True
    var_4 = False



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Test the loader function by mocking the Parser and walk_packages dependency.\n    We verify that the loader iterates through packages, attempts to parse\n    source/stubs, and handles extension modules correctly.\n    '
    var_1 = 'my_pkg'
    var_2 = '/fake/path/my_pkg'
    var_3 = (var_1, var_2)
    var_4 = '/fake/path'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_1, var_4, var_5, var_5, var_6)
    assert var_7 == 'compiled_doc'
    var_8 = "print('hello')"
    var_9 = 'ext_pkg'
    var_10 = '/fake/path/ext_pkg'
    var_11 = (var_9, var_10)
    var_12 = module_0.loader(var_9, var_4, var_5, var_5, var_6)
    assert var_12 == 'compiled_doc'
    var_13 = 'empty_pkg'
    var_14 = '/fake/path/empty_pkg'
    var_15 = (var_13, var_14)
    var_16 = module_0.loader(var_13, var_4, var_5, var_5, var_6)
    var_17 = 'no module for empty_pkg in this platform'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Test walk_packages by creating a mock filesystem structure and \n    verifying that it correctly identifies python files and ignores others.\n    '
    var_1 = 'pkg'
    var_2 = 'subpackage'
    var_3 = 'other_dir'
    var_4 = '__init__.py'
    var_5 = ''
    var_6 = 'module.py'
    var_7 = 'data.txt'
    var_8 = 'not a python file'
    var_9 = 'not_target.py'
    var_10 = '__init__.pyi'
    var_11 = 'pkg'
    var_12 = '.py'
    var_13 = '.pyi'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test walk_packages when no packages are found.'
    var_1 = '/tmp/root'
    var_2 = []
    var_3 = []
    var_4 = (var_1, var_2, var_3)
    var_5 = 'none'
    var_6 = module_0.walk_packages(var_5, var_1)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0

def test_case_0():
    var_0 = 'Test that PEP561_SUFFIX (-stubs) directories are handled correctly.'
    var_1 = 'my_pkg'
    var_2 = 'my_pkg-stubs'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = '__init__.pyi'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'my_package'
    var_1 = 'my_package.submodule'
    var_2 = 'submodule'
    var_3 = "__version__ = '1.0.0'"
    var_4 = '"""Submodule Docstring"""'
    var_5 = '"""Stub Docstring"""'
    var_6 = '/tmp'
    var_7 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the loader function.\n    Mocks Parser to verify it receives the correct calls during the walking process.\n    '
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = 'test_pkg.submodule'
    var_4 = 'submodule'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Tests that loader attempts to load extension modules when .py is not pure py.'
    var_1 = 'test_pkg.ext'
    var_2 = 'ext'
    var_3 = True
    var_4 = False



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the loader function by simulating a package walk and \n    verifying if the Parser receives the expected content.\n    '
    var_1 = 'test_pkg.submodule'
    var_2 = 'submodule'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'Tests that loader skips extension loading if a .py file is found (pure python).'
    var_1 = 'pure.py'
    var_2 = '"""Pure Py"""'
    var_3 = 'test_pkg.pure'
    var_4 = 'pure'
    var_5 = True
    var_6 = False



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function logic.'
    var_1 = 'my_package'
    var_2 = 'pkg_path'
    var_3 = True
    var_4 = 'root'
    var_5 = False

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that loader skips extension loading if a pure .py file is found.'
    var_1 = 'my_package'
    var_2 = 'pkg_path'
    var_3 = 'root'
    var_4 = 'pwd'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)
    assert var_7 == 'Pure Python Doc'
    var_8 = 'root'
    var_9 = 'pwd'
    var_10 = True
    var_11 = False
    var_12 = module_0.loader(var_8, var_9, var_10, var_10, var_11)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_pkg.module'
    var_1 = 'test_pkg/module'
    var_2 = 'test_pkg'
    var_3 = True
    var_4 = False

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when no packages are found.'
    var_1 = 'empty'
    var_2 = '/tmp'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the loader function by simulating a package walk.\n    It verifies that the parser is called and the final output is compiled.\n    '
    var_1 = 'test_pkg'
    var_2 = True
    var_3 = 'loading extension module for fully documented:'

def test_case_0():
    var_0 = 'Tests loader when no packages are found in the path.'
    var_1 = []
    var_2 = 'empty'
    var_3 = True
    var_4 = False



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests the loader function by mocking the package walking, \n    file reading, file existence checks, and module loading.\n    '
    var_1 = '/fake/root'
    var_2 = 'my_package'
    var_3 = '/fake/root/my_package'
    var_4 = '.so'
    var_5 = var_3 + var_4
    var_6 = (var_2, var_3)
    var_7 = True
    var_8 = module_0.loader(var_1, var_3, var_7, var_7, var_7)
    assert var_8 == 'Compiled Documentation'
    var_9 = 'stub content'

import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests that loader skips extension loading if a .py file is found (pure python).\n    '
    var_1 = 'pure_pkg'
    var_2 = '/fake/root/pure_pkg'
    var_3 = (var_1, var_2)
    var_4 = '.py'
    var_5 = '/fake'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_5, var_5, var_6, var_6, var_7)
    var_9 = "print('hello')"



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Tests the loader function logic.'
    var_1 = 'test_pkg.submodule'
    var_2 = 'submodule'
    var_3 = True



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test walk_packages by creating a temporary file structure.'
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = 'other'
    var_4 = ''
    var_5 = ''
    var_6 = ''
    var_7 = ''
    var_8 = ''
    var_9 = '.py'
    var_10 = '.pyi'
    var_11 = (var_9, var_10)
    var_12 = False
    var_13 = True

def test_case_0():
    var_0 = 'Test walk_packages when no packages match the criteria.'
    var_1 = 'other'
    var_2 = ''
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'Test walk_packages with an empty directory.'
    var_1 = 'pkg'



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests the loader function to ensure it iterates through packages, \n    parses .py and .pyi files, and attempts to load extension modules.\n    '
    var_1 = 'pkg_pure'
    var_2 = '/tmp/pkg_pure'
    var_3 = (var_1, var_2)
    var_4 = 'pkg_ext'
    var_5 = '/tmp/pkg_ext'
    var_6 = (var_4, var_5)
    var_7 = 'content of '
    var_8 = '/tmp'
    var_9 = True
    var_10 = module_0.loader(var_8, var_8, var_9, var_9, var_9)
    assert var_10 == 'compiled_doc'
    var_11 = 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when no packages are found.'
    var_1 = '/tmp'
    var_2 = True
    var_3 = module_0.loader(var_1, var_1, var_2, var_2, var_2)
    assert var_3 == 'compiled_doc'



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function by mocking file system and Parser behavior.'
    var_1 = '/fake/root'
    var_2 = '/fake/pwd'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = 'my_package'
    var_7 = '/fake/pwd/my_package'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_10 == '# Generated API Content'
    var_11 = '/fake/pwd/my_package.pyi'
    var_12 = '/fake/pwd/my_package.so'
    var_13 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_13 == '# Generated API Content'
    var_14 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_14 == '# Generated API Content'



# Parsed testcases at query #15
#--------------------------




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test walk_packages functionality including filtering and path resolution.'
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = 'other_pkg'
    var_4 = '__init__.py'
    var_5 = 'module1.py'
    var_6 = 'module2.pyi'
    var_7 = 'other.txt'
    var_8 = 'ignore me'
    var_9 = '.py'
    var_10 = '.pyi'
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'Test walk_packages with an empty directory.'
    var_1 = 'nonexistent'



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/fake_path'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_3, var_3, var_4)
    assert var_5 == 'Compiled Docstring'
    var_6 = module_0.loader(var_0, var_1, var_3, var_3, var_4)
    assert var_6 == 'Compiled Docstring'
    var_7 = 'loading extension module for fully documented:'
    var_8 = 'non_existent'
    var_9 = '/tmp/none'
    var_10 = module_0.loader(var_8, var_9, var_3, var_3, var_4)
    assert var_10 == 'Compiled Docstring'
    var_11 = 'empty_pkg'
    var_12 = '/tmp/empty'
    var_13 = (var_11, var_12)
    var_14 = module_0.loader(var_11, var_12, var_3, var_3, var_4)
    var_15 = 'no module for empty_pkg in this platform'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Test the loader function by mocking walk_packages, Parser, \n    and file system interactions.\n    '
    var_1 = 'test_pkg'
    var_2 = '/tmp/test'
    var_3 = (var_1, var_2)
    var_4 = '.py'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when no packages are found by walk_packages.'
    var_1 = 'empty'
    var_2 = '/tmp/empty'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'pkg'
    var_1 = 'subpkg'
    var_2 = 'other'
    var_3 = ''
    var_4 = ''
    var_5 = ''
    var_6 = ''
    var_7 = ''
    var_8 = 'hello'
    var_9 = 0

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = 'Test loader when no packages are found in the path.'
    var_1 = 'nonexistent'
    var_2 = True
    var_3 = False



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function logic.'
    var_1 = 'my_package'
    var_2 = '__init__'
    var_3 = True
    var_4 = 0
    var_5 = 'Should not attempt loading extensions if pure_py is True'

def test_case_0():
    var_0 = 'Test loader when it encounters a package that requires extension loading.'
    var_1 = 'ext_pkg'
    var_2 = '.__init__'
    var_3 = True
    var_4 = False



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == 'compiled_content'
    var_5 = 'pkg'
    var_6 = '/tmp/pkg'
    var_7 = (var_5, var_6)
    var_8 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_9 = 'content'
    var_10 = 'root'
    var_11 = 'pwd'
    var_12 = True
    var_13 = False
    var_14 = module_0.loader(var_10, var_11, var_12, var_12, var_13)
    var_15 = 'root'
    var_16 = 'pwd'
    var_17 = True
    var_18 = False
    var_19 = module_0.loader(var_15, var_16, var_17, var_17, var_18)
    var_20 = 'no module for pkg in this platform'



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Tests loader when no packages are found.'
    var_1 = 'non_existent'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_1, var_1, var_2, var_2, var_3)
    assert var_4 == ''



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function logic using mocks for filesystem and imports.'
    var_1 = 'my_package'
    var_2 = 'extension_module'
    var_3 = '.extension_module'
    var_4 = var_1 + var_3
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Test specifically the flow where an extension module is loaded.'
    var_1 = 'ext_pkg'
    var_2 = 'extension_module'
    var_3 = True
    var_4 = False



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Test walk_packages by creating a dummy directory structure and \n    verifying it correctly identifies Python files and stubs within the target package.\n    '
    var_1 = 'my_pkg'
    var_2 = '__init__.py'
    var_3 = 'content'
    var_4 = 'sub_mod'
    var_5 = 'module.py'
    var_6 = 'sub_mod-stubs'
    var_7 = 'module.pyi'
    var_8 = 'other_pkg'
    var_9 = 'other.py'
    var_10 = 'my_pkg'

def test_case_0():
    var_0 = 'Test walk_packages when no matching packages are found.'
    var_1 = 'empty'
    var_2 = 'non_existent'

def test_case_0():
    var_0 = '\n    Test the internal string manipulation logic of walk_packages \n    using a controlled mock of os.walk.\n    '
    var_1 = 'pkg'
    var_2 = 'sub'
    var_3 = [var_2]
    var_4 = '__init__.py'
    var_5 = [var_4]
    var_6 = 'pkg/sub'
    var_7 = []
    var_8 = 'module.py'
    var_9 = [var_8]
    var_10 = 'pkg'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'pkg_a'
    var_1 = 'pkg_b'
    var_2 = ''
    var_3 = ''
    var_4 = ''
    var_5 = ''
    var_6 = ''
    var_7 = 'module_b'
    var_8 = 'other'
    var_9 = 'pkg_a-stubs'
    var_10 = ''



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the loader function by mocking the Parser and walk_packages \n    to ensure it orchestrates the parsing of .py and .pyi files correctly.\n    '
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = 'test_pkg.submodule'
    var_4 = 'submodule'
    var_5 = '__init__'
    var_6 = "'''Root docstring'''"
    var_7 = "'''Stub docstring'''"
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Tests loader when no packages are found.'
    var_1 = True
    var_2 = False



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Test walk_packages by creating a dummy directory structure \n    containing valid and invalid files.\n    '
    var_1 = 'test_root'
    var_2 = 'pkg'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = 'module.py'
    var_6 = 'subpkg'
    var_7 = 'other.py'
    var_8 = 'stub.pyi'
    var_9 = 'nonexistent'

def test_case_0():
    var_0 = 'Ensure .txt or other files are not yielded.'
    var_1 = 'test_root'
    var_2 = 'pkg'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = 'readme.txt'
    var_6 = 'hello'
    var_7 = 'readme'

def test_case_0():
    var_0 = 'Ensure files ending with -stubs are treated correctly by the logic.'
    var_1 = 'test_root'
    var_2 = 'pkg'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = 'pkg-stubs'



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == 'compiled_module_doc'
    var_5 = 'mypkg'
    var_6 = '/tmp/mypkg'
    var_7 = (var_5, var_6)
    var_8 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_9 = "print('hello')"
    var_10 = 'mystub'
    var_11 = '/tmp/mystub'
    var_12 = (var_10, var_11)
    var_13 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_14 = '# stub'
    var_15 = 'badext'
    var_16 = '/tmp/badext'
    var_17 = (var_15, var_16)
    var_18 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_19 = 'no module for badext in this platform'



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Test the loader function by mocking the dependencies:\n    Parser, walk_packages, and file system checks.\n    '
    var_1 = '/fake/root'
    var_2 = '/site-packages/fake_pkg'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = 'my_package'
    var_7 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_7 == '# Generated API Content'
    var_8 = 'def func(): pass'
    var_9 = 'ext_pkg'
    var_10 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_10 == '# Generated API Content'
    var_11 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_11 == '# Generated API Content'
    var_12 = 'fail_pkg'
    var_13 = '.pyi'
    var_14 = '.pyd'
    var_15 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    var_16 = 'no module for fail_pkg in this platform'



