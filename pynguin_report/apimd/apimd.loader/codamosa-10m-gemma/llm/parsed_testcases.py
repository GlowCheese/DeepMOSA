####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg.submodule'
    var_2 = 'submodule'
    var_3 = True



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Unit test for the loader function.\n    Tests the orchestration of walking packages, parsing source/stubs,\n    and loading extension modules.\n    '
    var_1 = '/tmp/root'
    var_2 = '/tmp/pwd'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = 'my_pkg'
    var_7 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_7 == 'Compiled Docstring'
    var_8 = 'def hello(): pass'
    var_9 = 'ext_pkg'
    var_10 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_10 == 'Compiled Docstring'
    var_11 = '# Stub content'
    var_12 = 'broken_pkg'
    var_13 = '.pyi'
    var_14 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    var_15 = 'no module for broken_pkg in this platform'



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests the gen_api function by mocking filesystem operations, \n    module loading, and the parser/loader logic.\n    '
    var_1 = 'My Module'
    var_2 = 'my_module'
    var_3 = {var_1: var_2}
    var_4 = 'test_docs_output'
    var_5 = '/fake/pwd'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_3, var_5, prefix=var_4, link=var_6, level=var_6, toc=var_6, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'my-module-api.md'
    var_11 = module_0.gen_api(var_3, prefix=var_4, dry=var_6)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'This is the parsed docstring content.'
    var_14 = 'Empty'
    var_15 = 'empty_mod'
    var_16 = {var_14: var_15}
    var_17 = module_0.gen_api(var_16)
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = "'empty_mod' can not be found"



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Test gen_api function by mocking the underlying heavy-lifting functions:\n    loader, _site_path, _write, and os operations.\n    '
    var_1 = 'my_package'
    var_2 = 'My Package'
    var_3 = {var_1: var_2}
    var_4 = 'prefix'
    assert var_4 == 1
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = '## My Package API\n\nContents of the API docstring.'
    var_9 = 'my_package'
    var_10 = '/fake/path/to/package'
    var_11 = 'my-package-api.md'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with dry=True to ensure no files are written.'
    var_1 = 'pkg'
    var_2 = 'Pkg'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = False
    var_8 = True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when the loader returns an empty string (package not found).'
    var_1 = 'missing_pkg'
    var_2 = 'Missing'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = module_0.gen_api(var_3, prefix=var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = "'missing_pkg' can not be found"



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests the loader function by mocking the file system, \n    the parser, and the package walking process.\n    '
    var_1 = 'module'
    var_2 = 'test_pkg.module'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = module_0.loader(var_3, var_1, var_4, var_4, var_4)
    assert var_5 == 'Compiled Docstring'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = "\n    Test the loader function logic:\n    1. Iterates through packages.\n    2. Parses .py and .pyi files.\n    3. If it's an extension module (no .py), attempts to load via _load_module.\n    4. Returns compiled docstring.\n    "
    var_1 = 'test_pkg'
    var_2 = True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that loader skips extension loading if a .py file is found (pure python).'
    var_1 = 'pure_pkg'
    var_2 = '/tmp/pure_pkg'
    var_3 = (var_1, var_2)
    var_4 = '/tmp/pure_pkg'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_4, var_5, var_5, var_6)
    assert var_7 == 'Pure Py Doc'



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with dry=True to ensure no files are written and logs are printed.'
    var_1 = 'MyModule'
    var_2 = 'my_module'
    var_3 = {var_1: var_2}
    var_4 = 'test_docs_dry'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = False
    var_8 = True
    var_9 = len(var_6)
    assert var_9 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns an empty string (module not found).'
    var_1 = 'EmptyModule'
    var_2 = 'empty_mod'
    var_3 = {var_1: var_2}
    var_4 = 'test_empty'
    var_5 = module_0.gen_api(var_3, prefix=var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = "'empty_mod' can not be found"



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Test gen_api functionality including directory creation, \n    module loading, and file writing.\n    '
    var_1 = 'prefix'
    var_2 = 'root_names'
    var_3 = '/fake/pwd'
    var_4 = True
    var_5 = False
    var_6 = '# Test Package API\n\nThis is a docstring content.'
    var_7 = 'test-pkg-api.md'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with dry=True to ensure no files are written.'
    var_1 = 'pkg'
    var_2 = 'Pkg'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = 'docs'
    var_6 = module_0.gen_api(var_3, prefix=var_5, dry=var_4)
    var_7 = '='
    var_8 = 12
    var_9 = var_7 * var_8
    var_10 = False
    var_11 = True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty string (package not found).'
    var_1 = 'missing_pkg'
    var_2 = 'Missing'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = module_0.gen_api(var_3, prefix=var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = "'missing_pkg' can not be found"



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'MyModule'
    var_1 = 'my_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, link=var_4, level=var_4, toc=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'my-module-api.md'
    var_8 = '# MyModule API\n\nGenerated Docstring Content'
    var_9 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = '='
    var_12 = 12
    var_13 = var_11 * var_12
    var_14 = module_0.gen_api(var_2, prefix=var_3)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = "'my_module' can not be found"
    var_17 = 'Test'
    var_18 = 'test_pkg'
    var_19 = {var_17: var_18}
    var_20 = 'new_dir'
    var_21 = module_0.gen_api(var_19, prefix=var_20)
    var_22 = 'Create directory: new_dir'
    var_23 = 'A'
    var_24 = 'a'
    var_25 = {var_23: var_24}
    var_26 = '/fake/pwd'
    var_27 = module_0.gen_api(var_25, var_26)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg.module'
    var_2 = '/module'
    var_3 = 'test_pkg'
    var_4 = True

import apimd.loader as module_0

def test_case_0():
    var_0 = "Test that if a .py file exists, it doesn't attempt to load extensions."
    var_1 = 'pure_pkg'
    var_2 = '__init__.py'
    var_3 = 'pass'
    var_4 = 'pure_pkg'
    var_5 = (var_4, var_1)
    var_6 = 'pure_pkg'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_6, var_1, var_7, var_7, var_8)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'pkg'
    var_1 = 'subpkg'
    var_2 = '__init__.py'
    var_3 = 'module.py'
    var_4 = 'submodule.pyi'
    var_5 = 'other.txt'
    var_6 = ''
    var_7 = ''
    var_8 = ''
    var_9 = ''
    var_10 = ''
    var_11 = 'nonexistent'
    var_12 = 'subpkg-stubs'
    var_13 = '__init__.pyi'
    var_14 = ''



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with dry=True to ensure it logs instead of writing.'
    var_1 = 'pkg'
    var_2 = 'Pkg'
    var_3 = {var_1: var_2}
    var_4 = 'dry_test'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = 'content'
    var_8 = len(var_6)
    assert var_8 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when the loader returns an empty string.'
    var_1 = 'empty_pkg'
    var_2 = 'Empty'
    var_3 = {var_1: var_2}
    var_4 = 'empty_test'
    var_5 = module_0.gen_api(var_3, prefix=var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = "'empty_pkg' can not be found"



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Test the loader function by mocking the Parser and the walk_packages iterator.\n    '
    var_1 = 'my_package'
    var_2 = '/fake/path/my_package'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = 'my_package'
    var_7 = '/fake/path'
    var_8 = True
    var_9 = module_0.loader(var_6, var_7, var_8, var_8, var_8)
    assert var_9 == 'Compiled Documentation'
    var_10 = module_0.loader(var_6, var_7, var_8, var_8, var_8)
    assert var_10 == 'Compiled Documentation'
    var_11 = module_0.loader(var_6, var_7, var_8, var_8, var_8)
    assert var_11 == 'Compiled Documentation'



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'MyModule'
    var_1 = 'OtherPkg'
    var_2 = 'my_module'
    var_3 = 'other_pkg'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_docs'
    var_6 = '/fake/pwd'
    var_7 = True
    var_8 = 2
    var_9 = False
    var_10 = module_0.gen_api(var_4, var_6, prefix=var_5, link=var_7, level=var_8, toc=var_7, dry=var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'my-module-api.md'
    var_13 = 'other-pkg-api.md'
    var_14 = 'my_module'
    var_15 = {var_14: var_14}
    var_16 = module_0.gen_api(var_15, prefix=var_5, dry=var_7)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = '='
    var_19 = 12
    var_20 = var_18 * var_19
    var_21 = 'a'
    var_22 = {var_21: var_21}
    var_23 = 'new_dir'
    var_24 = module_0.gen_api(var_22, prefix=var_23)
    var_25 = 'missing'
    var_26 = {var_25: var_25}
    var_27 = module_0.gen_api(var_26, prefix=var_5)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = "'missing' can not be found"



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function with mocked Parser and walk_packages.'
    var_1 = 'pkg_name'
    var_2 = 'root'
    var_3 = '__init__.py'
    var_4 = 'sub_name'
    var_5 = 'submodule'
    var_6 = '__init__.pyi'
    var_7 = 'ext_name'
    var_8 = 'ext_mod'
    var_9 = True

def test_case_0():
    var_0 = 'Test that loader skips extension loading if a pure .py file is found.'
    var_1 = 'pkg_name'
    var_2 = 'root'
    var_3 = '__init__.py'
    var_4 = True
    var_5 = False



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Unit test for the loader function.\n    Tests the orchestration of walking packages, parsing files, \n    and loading extension modules.\n    '
    var_1 = '/fake/root'
    var_2 = '/fake/pwd'
    var_3 = True
    var_4 = 1
    var_5 = True
    var_6 = 'my_package'
    var_7 = '/fake/pwd/my_package'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = module_0.loader(var_1, var_2, var_3, var_4, var_5)
    assert var_10 == 'Compiled Docstring'
    var_11 = 'stub content'

import apimd.loader as module_0

def test_case_0():
    var_0 = '\n    Tests that if a .py file is found, the loader skips extension loading.\n    '
    var_1 = 'pure_pkg'
    var_2 = '/fake/pwd/pure_pkg'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = 'root'
    var_7 = 'pwd'
    var_8 = True
    var_9 = False
    var_10 = module_0.loader(var_6, var_7, var_8, var_8, var_9)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the gen_api function by mocking the heavy lifting (loader and site_path)\n    to verify the orchestration logic: directory creation, file writing, and return values.\n    '
    var_1 = 'docs_output'
    var_2 = 'my_module'
    var_3 = 'other_pkg'
    var_4 = 'My Module Title'
    assert var_4 == 2
    assert var_4 == 0
    var_5 = 'Other Package'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Generated Docstring Content'
    var_8 = True
    var_9 = 2
    var_10 = False
    var_11 = 'my-module-api.md'
    var_12 = 'other-pkg-api.md'
    var_13 = '='
    var_14 = 12
    var_15 = var_13 * var_14
    var_16 = 'missing_pkg'
    var_17 = 'Missing'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = "'missing_pkg' can not be found"



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function with mocked Parser and filesystem.'
    var_1 = 'test_pkg.submodule'
    var_2 = 'test_pkg/submodule'
    var_3 = 'test_pkg.real_sub'
    var_4 = 'test_pkg/real_sub'
    var_5 = 'test_pkg'
    var_6 = True
    var_7 = 'dummy content'

def test_case_0():
    var_0 = 'Test that if a module is pure .py, it skips extension loading.'
    var_1 = 'test_pkg.real_sub'
    var_2 = 'test_pkg/real_sub'
    var_3 = 'test_pkg'
    var_4 = True



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = 'my_package.submodule'
    var_4 = '/tmp/test_pkg/sub'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'root'
    var_8 = '/tmp/test_pkg'
    var_9 = True
    var_10 = False
    var_11 = module_0.loader(var_7, var_8, var_9, var_9, var_10)
    assert var_11 == 'compiled_doc_string'
    var_12 = module_0.loader(var_7, var_8, var_9, var_9, var_10)
    assert var_12 == 'compiled_doc_string'
    var_13 = 'loading extension module for fully documented:'
    var_14 = 'root'
    var_15 = '/tmp/ext'
    var_16 = True
    var_17 = False
    var_18 = module_0.loader(var_14, var_15, var_16, var_16, var_17)
    var_19 = 'no module for ext_pkg in this platform'



