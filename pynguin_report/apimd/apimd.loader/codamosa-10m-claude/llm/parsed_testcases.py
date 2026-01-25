####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '\n"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '\n"""Stub module docstring."""\n\ndef stub_func() -> None: ...\n'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with different header levels.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module."""\ndef func(): """Func."""'
    var_6 = 'test_pkg'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module."""'
    var_6 = True

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module."""'
    var_6 = False
    var_7 = 1

def test_case_0():
    var_0 = 'Test loader with non-existent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '\n"""Module docstring."""\n\ndef func():\n    """Function docstring."""\n    pass\n\nclass MyClass:\n    """Class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '\n"""Stub module docstring."""\n\ndef stub_func() -> None: ...\n'
    var_8 = 'PYTHONPATH'
    var_9 = True
    var_10 = False
    var_11 = 'module'

def test_case_0():
    var_0 = 'Test loader with different link and level parameters.'
    var_1 = 'test_pkg2'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '\n"""Test module."""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_6 = 'PYTHONPATH'
    var_7 = False
    var_8 = 2

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg3'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'documented.py'
    var_5 = '\n"""Documented module."""\n\ndef documented_func():\n    """A documented function."""\n    pass\n'
    var_6 = 'PYTHONPATH'
    var_7 = True

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'PYTHONPATH'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with non-existent path.'
    var_1 = 'nonexistent'
    var_2 = True
    var_3 = False



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module docstring."""\ndef func():\n    """Function docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '"""Stub module docstring."""\ndef stub_func() -> None: ...\n'
    var_8 = True
    var_9 = False
    var_10 = 2
    var_11 = 'nonexistent_pkg'



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.sub'
    var_5 = '/path/to/test_module/sub'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = "\ndef test_function():\n    '''Test function documentation.'''\n    pass\n\nclass TestClass:\n    '''Test class documentation.'''\n    pass\n"
    var_9 = 'test_package'
    var_10 = '/test/path'
    var_11 = True
    var_12 = False
    var_13 = module_0.loader(var_9, var_10, var_11, var_11, var_12)
    assert var_13 == '# Compiled Documentation\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with extension modules.'
    var_1 = 'extension_module'
    var_2 = '/path/to/extension_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'ext_pkg'
    var_6 = '/test/path'
    var_7 = False
    var_8 = 2
    var_9 = True
    var_10 = module_0.loader(var_5, var_6, var_7, var_8, var_9)
    assert var_10 == '# Extension Module Docs\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function when no packages are found.'
    var_1 = 'nonexistent'
    var_2 = '/test/path'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = '/test/path'
    var_5 = 'docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_3, var_4, prefix=var_5, link=var_6, level=var_6, toc=var_7, dry=var_6)
    var_9 = len(var_8)
    assert var_9 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api without pwd parameter.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = "Test gen_api creates directory when it doesn't exist."
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api handles empty documentation.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api writes file when dry=False.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple modules.'
    var_1 = 'Test1'
    var_2 = 'Test2'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = False
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = 'Test'
    var_2 = 'test_module_name'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = True
    var_6 = module_0.gen_api(var_3, level=var_4, dry=var_5)



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = {}
    var_2 = module_0.gen_api(var_1)
    var_3 = 'site_packages'
    var_4 = 'Test'
    var_5 = 'test_module'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = 'docs'
    var_9 = module_0.gen_api(var_6, prefix=var_8, dry=var_7)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'site_packages'
    var_12 = 'MyModule'
    var_13 = 'my_module'
    var_14 = {var_12: var_13}
    var_15 = False
    var_16 = 'docs'
    var_17 = module_0.gen_api(var_14, prefix=var_16, dry=var_15)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'site_packages'
    var_20 = 'Mod1'
    var_21 = 'Mod2'
    var_22 = 'module1'
    var_23 = 'module2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = True
    var_26 = module_0.gen_api(var_24, dry=var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = 'Empty'
    var_29 = 'empty_module'
    var_30 = {var_28: var_29}
    var_31 = True
    var_32 = module_0.gen_api(var_30, dry=var_31)
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = 'site_packages'
    var_35 = 'Test'
    var_36 = 'test'
    var_37 = {var_35: var_36}
    var_38 = '/custom/path'
    var_39 = True
    var_40 = module_0.gen_api(var_37, var_38, dry=var_39)
    var_41 = 'site_packages'
    var_42 = 'Test'
    var_43 = 'test'
    var_44 = {var_42: var_43}
    var_45 = 'custom_docs'
    var_46 = 3
    var_47 = True
    var_48 = module_0.gen_api(var_44, prefix=var_45, level=var_46, dry=var_47)
    var_49 = len(var_48)
    assert var_49 == 1



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function with various scenarios.'
    var_1 = 'Module1'
    var_2 = 'module1'
    var_3 = {var_1: var_2}
    var_4 = '/custom/path'
    var_5 = 'docs'
    var_6 = True
    var_7 = module_0.gen_api(var_3, var_4, prefix=var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'Module1'
    var_10 = 'module1'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = '## Mod1\nDoc1'
    var_15 = '## Mod2\nDoc2'
    var_16 = 'Title1'
    var_17 = 'Title2'
    var_18 = 'module1'
    var_19 = 'module2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = False
    var_22 = module_0.gen_api(var_20, dry=var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'Module'
    var_25 = 'module'
    var_26 = {var_24: var_25}
    var_27 = 'new_docs'
    var_28 = module_0.gen_api(var_26, prefix=var_27)
    var_29 = 'Module'
    var_30 = 'my_module_name'
    var_31 = {var_29: var_30}
    var_32 = 'docs'
    var_33 = False
    var_34 = module_0.gen_api(var_31, prefix=var_32, dry=var_33)
    var_35 = 'Title'
    var_36 = 'module'
    var_37 = {var_35: var_36}
    var_38 = 3
    var_39 = True
    var_40 = module_0.gen_api(var_37, level=var_38, dry=var_39)
    var_41 = 'Module'
    var_42 = 'module'
    var_43 = {var_41: var_42}
    var_44 = None
    var_45 = module_0.gen_api(var_43, var_44)
    var_46 = 'Module'
    var_47 = 'module'
    var_48 = {var_46: var_47}
    var_49 = False
    var_50 = 2
    var_51 = True
    var_52 = module_0.gen_api(var_48, link=var_49, level=var_50, toc=var_51)



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = {}
    var_2 = module_0.gen_api(var_1)
    var_3 = '/fake/path'
    var_4 = [var_3]
    var_5 = 'TestModule'
    var_6 = 'test_module'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7)
    var_9 = '/custom/path'
    var_10 = module_0.gen_api(var_7, var_9)
    var_11 = 'custom_docs'
    var_12 = module_0.gen_api(var_7, prefix=var_11)
    var_13 = True
    var_14 = module_0.gen_api(var_7, dry=var_13)
    var_15 = 'api_docs'
    var_16 = False
    var_17 = 2
    var_18 = module_0.gen_api(var_7, prefix=var_15, link=var_16, level=var_17, toc=var_13, dry=var_16)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_4 = 'submodule.py'
    var_5 = '"""Submodule."""\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '"""Stub module."""\ndef stub_func() -> None: ...\n'
    var_8 = 'sys.path'
    var_9 = True
    var_10 = False

def test_case_0():
    var_0 = 'Test the loader function with table of contents.'
    var_1 = 'test_pkg_toc'
    var_2 = '__init__.py'
    var_3 = '"""Test package with TOC."""\n'
    var_4 = 'sys.path'
    var_5 = True

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Level test package."""\n'
    var_4 = 'sys.path'
    var_5 = 'level_test_pkg'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'no_link_pkg'
    var_2 = '__init__.py'
    var_3 = '"""No link package."""\n'
    var_4 = 'sys.path'
    var_5 = False
    var_6 = 1



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function with mocked dependencies.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\nclass TestClass:\n    """Test class."""\n    pass'
    var_6 = 'test_pkg'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = 'def stub_func() -> None: ...'
    var_4 = 'stub_pkg'
    var_5 = False
    var_6 = 2
    var_7 = True

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = 'empty_pkg'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'Test loader attempts to load extension modules.'
    var_1 = 'ext_pkg'
    var_2 = 'module.pyi'
    var_3 = 'def ext_func() -> None: ...'
    var_4 = 'ext_pkg'
    var_5 = True
    var_6 = False



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function with various scenarios.'
    var_1 = 'Test Module'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = '/test/path'
    var_5 = 'docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_3, var_4, prefix=var_5, link=var_6, level=var_6, toc=var_7, dry=var_6)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'Test Module'
    var_11 = 'test_module'
    var_12 = {var_10: var_11}
    var_13 = 'custom_docs'
    var_14 = True
    var_15 = module_0.gen_api(var_12, prefix=var_13, dry=var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = '# Module1 API\n\nDoc1'
    var_18 = '# Module2 API\n\nDoc2'
    var_19 = 'Module One'
    var_20 = 'Module Two'
    var_21 = 'module1'
    var_22 = 'module2'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = True
    var_25 = module_0.gen_api(var_23, dry=var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'Missing Module'
    var_28 = 'missing'
    var_29 = {var_27: var_28}
    var_30 = True
    var_31 = module_0.gen_api(var_29, dry=var_30)
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = 'Test Module'
    var_34 = 'test_module'
    var_35 = {var_33: var_34}
    var_36 = 'docs'
    var_37 = False
    var_38 = module_0.gen_api(var_35, prefix=var_36, dry=var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = 'Test Module'
    var_41 = 'test_module'
    var_42 = {var_40: var_41}
    var_43 = 2
    var_44 = True
    var_45 = module_0.gen_api(var_42, level=var_43, dry=var_44)
    var_46 = 'Test Module'
    var_47 = 'test_module'
    var_48 = {var_46: var_47}
    var_49 = None
    var_50 = True
    var_51 = module_0.gen_api(var_48, var_49, dry=var_50)
    var_52 = len(var_51)
    assert var_52 == 1



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = '/site/pkg'
    var_2 = [var_1]
    var_3 = 'TestAPI'
    var_4 = 'test_pkg'
    var_5 = {var_3: var_4}
    var_6 = '/custom/path'
    var_7 = 'docs'
    var_8 = True
    var_9 = module_0.gen_api(var_5, var_6, prefix=var_7, dry=var_8)
    var_10 = '/site/pkg'
    var_11 = [var_10]
    var_12 = 'API'
    var_13 = 'pkg'
    var_14 = {var_12: var_13}
    var_15 = 'new_docs'
    var_16 = module_0.gen_api(var_14, prefix=var_15)
    var_17 = '/site/pkg'
    var_18 = [var_17]
    var_19 = 'API1'
    var_20 = 'API2'
    var_21 = 'pkg1'
    var_22 = 'pkg2'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = True
    var_25 = module_0.gen_api(var_23, dry=var_24)
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '/site/pkg'
    var_28 = [var_27]
    var_29 = 'API'
    var_30 = 'pkg'
    var_31 = {var_29: var_30}
    var_32 = None
    var_33 = True
    var_34 = module_0.gen_api(var_31, var_32, dry=var_33)
    var_35 = '/site/pkg'
    var_36 = [var_35]
    var_37 = 'API'
    var_38 = 'nonexistent_pkg'
    var_39 = {var_37: var_38}
    var_40 = True
    var_41 = module_0.gen_api(var_39, dry=var_40)
    var_42 = isinstance(var_41, var_33)
    var_43 = '/site/pkg'
    var_44 = [var_43]
    var_45 = 'CustomAPI'
    var_46 = 'my_pkg'
    var_47 = {var_45: var_46}
    var_48 = 'custom_docs'
    var_49 = False
    var_50 = 2
    var_51 = True
    var_52 = module_0.gen_api(var_47, prefix=var_48, link=var_49, level=var_50, toc=var_51, dry=var_51)



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.submodule'
    var_5 = '/path/to/test_module/submodule'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass'
    var_9 = '"""Test stub file."""\ndef test_func() -> None: ...'
    var_10 = '.py'
    var_11 = 'test_module'
    var_12 = '/path/to'
    var_13 = True
    var_14 = False
    var_15 = module_0.loader(var_11, var_12, var_13, var_13, var_14)
    var_16 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with stub files.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '"""Python source."""'
    var_6 = '"""Stub file."""'
    var_7 = True
    var_8 = 'test_module'
    var_9 = '/path/to'
    var_10 = False
    var_11 = 2
    var_12 = module_0.loader(var_8, var_9, var_10, var_11, var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function when no packages are found.'
    var_1 = 'nonexistent'
    var_2 = '/path/to'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with extension modules.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = True
    var_7 = 'test_module'
    var_8 = '/path/to'
    var_9 = True
    var_10 = False
    var_11 = module_0.loader(var_7, var_8, var_9, var_9, var_10)



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with mocked dependencies.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.submodule'
    var_5 = '/path/to/test_module/submodule'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 'test_root'
    var_9 = '/path/to/pwd'
    var_10 = True
    var_11 = False
    var_12 = module_0.loader(var_8, var_9, var_10, var_10, var_11)
    assert var_12 == '# Test Documentation\n\nTest content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with extension modules.'
    var_1 = 'test_extension'
    var_2 = '/path/to/test_extension'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'test_root'
    var_6 = '/path/to/pwd'
    var_7 = False
    var_8 = 2
    var_9 = True
    var_10 = module_0.loader(var_5, var_6, var_7, var_8, var_9)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with empty package list.'
    var_1 = 'empty_root'
    var_2 = '/path/to/pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'Test Module'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'My Module'
    var_8 = 'my_module'
    var_9 = {var_7: var_8}
    var_10 = 'docs'
    var_11 = False
    var_12 = module_0.gen_api(var_9, prefix=var_10, dry=var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'Test'
    var_15 = 'test'
    var_16 = {var_14: var_15}
    var_17 = '/custom/path'
    var_18 = True
    var_19 = module_0.gen_api(var_16, var_17, dry=var_18)
    var_20 = 'Missing'
    var_21 = 'missing_module'
    var_22 = {var_20: var_21}
    var_23 = True
    var_24 = module_0.gen_api(var_22, dry=var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = 'Test'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = 'custom_docs'
    var_30 = True
    var_31 = module_0.gen_api(var_28, prefix=var_29, dry=var_30)
    var_32 = 'Module A'
    var_33 = 'Module B'
    var_34 = 'Module C'
    var_35 = 'mod_a'
    var_36 = 'mod_b'
    var_37 = 'mod_c'
    var_38 = {var_32: var_35, var_33: var_36, var_34: var_37}
    var_39 = True
    var_40 = module_0.gen_api(var_38, dry=var_39)
    var_41 = len(var_40)
    assert var_41 == 3



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '"""Stub module."""\n\ndef stub_function() -> None: ...\n'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with empty directory.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with different parameters.'
    var_1 = 'param_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Module."""\ndef func(): pass\n'
    var_6 = False
    var_7 = 1
    var_8 = True
    var_9 = 2
    var_10 = True
    var_11 = True

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False



# Parsed testcases at query #2
#--------------------------


import genericpath as module_0

def test_case_0():
    var_0 = 'Test walk_packages function.'
    var_1 = 'test_package'
    var_2 = '__init__.py'
    var_3 = '# init'
    var_4 = 'module1.py'
    var_5 = '# module1'
    var_6 = 'module2.pyi'
    var_7 = '# module2 stub'
    var_8 = 'test_package-stubs'
    var_9 = '__init__.pyi'
    var_10 = '# stub init'
    var_11 = 'module3.pyi'
    var_12 = '# module3 stub'
    var_13 = 'subpkg'
    var_14 = '# subpkg init'
    var_15 = 'submodule.py'
    var_16 = '# submodule'
    var_17 = module_0.isdir(var_0)
    var_18 = '.py'
    var_19 = module_0.isfile(var_2)
    var_20 = '.pyi'

def test_case_0():
    var_0 = 'Test walk_packages with empty directory.'
    var_1 = 'empty_pkg'

def test_case_0():
    var_0 = 'Test walk_packages ignores non-python files.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '# init'
    var_4 = 'module.py'
    var_5 = '# module'
    var_6 = 'readme.txt'
    var_7 = '# readme'
    var_8 = 'data.json'
    var_9 = '{}'

def test_case_0():
    var_0 = 'Test walk_packages with deeply nested packages.'
    var_1 = 'root_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'level1'
    var_5 = 'level2'
    var_6 = 'deep_module.py'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '\n"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '\n"""Stub module docstring."""\n\ndef stub_func() -> None: ...\n'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module doc."""\n'
    var_6 = True
    var_7 = False
    var_8 = 2
    var_9 = 3

def test_case_0():
    var_0 = 'Test loader with link parameter variations.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module."""\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with table of contents parameter.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '"""Module."""\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test walk_packages function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '# init'
    var_4 = 'module1.py'
    var_5 = '# module1'
    var_6 = 'module2.pyi'
    var_7 = '# module2 stub'
    var_8 = 'subpkg'
    var_9 = '# sub init'
    var_10 = 'submodule.py'
    var_11 = '# submodule'
    var_12 = 'test_pkg-stubs'
    var_13 = '__init__.pyi'
    var_14 = '# stub init'
    var_15 = 'stub_module.pyi'
    var_16 = '# stub module'
    var_17 = -1
    var_18 = '.'
    var_19 = name.split(var_18)[var_17]

def test_case_0():
    var_0 = 'Test walk_packages with empty directory.'
    var_1 = 'empty_pkg'

def test_case_0():
    var_0 = 'Test walk_packages ignores non-Python files.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '# init'
    var_4 = 'module.py'
    var_5 = '# module'
    var_6 = 'readme.txt'
    var_7 = '# readme'
    var_8 = 'data.json'
    var_9 = '{}'

def test_case_0():
    var_0 = 'Test walk_packages removes PEP 561 suffix from names.'
    var_1 = 'test_pkg-stubs'
    var_2 = '__init__.pyi'
    var_3 = '# stub init'
    var_4 = 'module.pyi'
    var_5 = '# stub module'
    var_6 = 'test_pkg'

def test_case_0():
    var_0 = 'Test walk_packages with deeply nested packages.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '# init'
    var_4 = 'level1'
    var_5 = '# level1'
    var_6 = 'level2'
    var_7 = '# level2'
    var_8 = 'deep_module.py'
    var_9 = '# deep'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = 'packages'
    var_3 = 'test_module'
    var_4 = '__init__.py'
    var_5 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_6 = 'os.path.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'importlib.util.find_spec'
    var_10 = 'obj'
    var_11 = 'submodule_search_locations'
    var_12 = 'Test Module'
    var_13 = {var_12: var_3}
    var_14 = False
    var_15 = 'test-module-api.md'

def test_case_0():
    var_0 = 'Test gen_api with dry run.'
    var_1 = 'docs'
    var_2 = 'packages'
    var_3 = 'test_module'
    var_4 = '__init__.py'
    var_5 = '"""Test module."""\n'
    var_6 = 'os.path.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'importlib.util.find_spec'
    var_10 = 'obj'
    var_11 = 'submodule_search_locations'
    var_12 = 'Test'
    var_13 = {var_12: var_3}
    var_14 = 'test-module-api.md'

def test_case_0():
    var_0 = "Test that gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'packages'
    var_3 = 'test_module'
    var_4 = '__init__.py'
    var_5 = '"""Test."""\n'
    var_6 = 'os.path.isdir'
    var_7 = 'importlib.util.find_spec'
    var_8 = 'obj'
    var_9 = 'submodule_search_locations'
    var_10 = 'Test'
    var_11 = {var_10: var_3}
    var_12 = False

def test_case_0():
    var_0 = 'Test gen_api with non-existent module.'
    var_1 = 'docs'
    var_2 = 'os.path.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'importlib.util.find_spec'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'Missing'
    var_9 = 'nonexistent_module'
    var_10 = {var_8: var_9}

def test_case_0():
    var_0 = 'Test gen_api with custom level and toc settings.'
    var_1 = 'docs'
    var_2 = 'packages'
    var_3 = 'test_module'
    var_4 = '__init__.py'
    var_5 = '"""Test module."""\n'
    var_6 = 'os.path.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'importlib.util.find_spec'
    var_10 = 'obj'
    var_11 = 'submodule_search_locations'
    var_12 = 'Test'
    var_13 = {var_12: var_3}
    var_14 = False
    var_15 = 3



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.submodule'
    var_5 = '/path/to/test_module/submodule'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = '\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_9 = '\ndef test_function() -> None: ...\nclass TestClass: ...\n'
    var_10 = 'test_root'
    var_11 = '/path/to/pwd'
    var_12 = True
    var_13 = False
    var_14 = module_0.loader(var_10, var_11, var_12, var_12, var_13)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with extension modules.'
    var_1 = 'native_module'
    var_2 = '/path/to/native_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'test_root'
    var_6 = '/path/to/pwd'
    var_7 = False
    var_8 = 2
    var_9 = True
    var_10 = module_0.loader(var_5, var_6, var_7, var_8, var_9)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with no packages.'
    var_1 = 'empty_root'
    var_2 = '/path/to/pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function prioritizing stub files.'
    var_1 = 'stub_module'
    var_2 = '/path/to/stub_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'def stub_func() -> int: ...'
    var_6 = 'stub_root'
    var_7 = '/path/to/pwd'
    var_8 = True
    var_9 = False
    var_10 = module_0.loader(var_6, var_7, var_8, var_8, var_9)
    var_11 = 'stub_module'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = 'site-packages'
    var_3 = '## Class Example\n\nExample class documentation.'
    var_4 = 'Example'
    var_5 = 'example_module'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test gen_api with multiple root packages.'
    var_1 = '## Module 1'
    var_2 = '## Module 2'
    var_3 = 'First'
    var_4 = 'Second'
    var_5 = 'module1'
    var_6 = 'module2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api when documentation is empty.'
    var_1 = 'Empty'
    var_2 = 'empty_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with dry run enabled.'
    var_1 = '## Test Documentation'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = 'docs'
    var_6 = True

def test_case_0():
    var_0 = 'Test gen_api with different heading levels and table of contents.'
    var_1 = '## Class Example'
    var_2 = 'Custom'
    var_3 = 'custom_module'
    var_4 = {var_2: var_3}
    var_5 = 'docs'
    var_6 = 3
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'Test that gen_api appends pwd to sys.path.'
    var_1 = 'custom_path'
    var_2 = '## Documentation'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = 'docs'

def test_case_0():
    var_0 = 'Test that underscores in module names are converted to hyphens in filenames.'
    var_1 = '## Documentation'
    var_2 = 'Example'
    var_3 = 'example_module_name'
    var_4 = {var_2: var_3}
    var_5 = 'docs'



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'Test Module'
    var_2 = 'Another'
    var_3 = 'test_module'
    var_4 = 'another_mod'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'Test'
    var_2 = 'test_mod'
    var_3 = {var_1: var_2}
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = []
    var_2 = iter(var_1)
    var_3 = 'Test'
    var_4 = 'test_mod'
    var_5 = {var_3: var_4}
    var_6 = '/custom/path'
    var_7 = True
    var_8 = module_0.gen_api(var_5, var_6, dry=var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api writes files when dry=False.'
    var_1 = 'My Module'
    var_2 = 'my_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api skips modules with empty documentation.'
    var_1 = '   \n  '
    var_2 = '## Valid'
    var_3 = 'Empty'
    var_4 = 'Valid'
    var_5 = 'empty_mod'
    var_6 = 'valid_mod'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = True
    var_9 = module_0.gen_api(var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with different heading levels.'
    var_1 = 'Test'
    var_2 = 'test_mod'
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = True
    var_6 = module_0.gen_api(var_3, level=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = 'Test'
    var_2 = 'test_module_name'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'another_module'
    var_5 = '/path/to/another_module'
    var_6 = (var_4, var_5)
    var_7 = 'test_root'
    var_8 = '/pwd'
    var_9 = True
    var_10 = False
    var_11 = module_0.loader(var_7, var_8, var_9, var_9, var_10)
    assert var_11 == 'compiled_output'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with extension modules.'
    var_1 = 'ext_module'
    var_2 = '/path/to/ext_module'
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = True
    var_6 = 'test_root'
    var_7 = '/pwd'
    var_8 = 2
    var_9 = module_0.loader(var_6, var_7, var_4, var_8, var_5)
    assert var_9 == 'compiled_with_extension'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when no module can be loaded.'
    var_1 = 'missing_module'
    var_2 = '/path/to/missing'
    var_3 = (var_1, var_2)
    var_4 = 'test_root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'empty_output'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '\n"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '\n"""Stub module docstring."""\n\ndef stub_func() -> None: ...\n'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '\n"""Package docstring."""\n'
    var_4 = True
    var_5 = False
    var_6 = 2

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'toc_pkg'
    var_2 = '__init__.py'
    var_3 = '\n"""Package with TOC."""\n'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = '\n"""Package without links."""\n'
    var_4 = False
    var_5 = 1



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = 'pyslvs.compiler._site_path'
    var_4 = 'pyslvs.compiler._write'
    var_5 = 'pyslvs.compiler.isdir'
    var_6 = 'MyPackage'
    var_7 = 'my_package'
    var_8 = {var_6: var_7}
    var_9 = False
    var_10 = '/fake/site/path'
    var_11 = True

def test_case_0():
    var_0 = 'Test gen_api with dry run.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = 'pyslvs.compiler._site_path'
    var_4 = 'pyslvs.compiler._write'
    var_5 = 'pyslvs.compiler.isdir'
    var_6 = 'Test'
    var_7 = 'test_pkg'
    var_8 = {var_6: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = 'pyslvs.compiler._site_path'
    var_4 = 'pyslvs.compiler._write'
    var_5 = 'pyslvs.compiler.isdir'
    var_6 = 'Empty'
    var_7 = 'empty_pkg'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = '# Package 1'
    var_4 = '# Package 2'
    var_5 = 'pyslvs.compiler._site_path'
    var_6 = 'pyslvs.compiler._write'
    var_7 = 'pyslvs.compiler.isdir'
    var_8 = 'Pkg1'
    var_9 = 'Pkg2'
    var_10 = 'pkg1'
    var_11 = 'pkg2'
    var_12 = {var_8: var_10, var_9: var_11}

def test_case_0():
    var_0 = 'Test gen_api with custom parameters.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = 'pyslvs.compiler._site_path'
    var_4 = 'pyslvs.compiler._write'
    var_5 = 'pyslvs.compiler.isdir'
    var_6 = 'Title'
    var_7 = 'module'
    var_8 = {var_6: var_7}
    var_9 = '/custom/pwd'
    var_10 = False
    var_11 = 2
    var_12 = True
    var_13 = '/path'

def test_case_0():
    var_0 = "Test that gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'pyslvs.compiler.logger'
    var_3 = 'pyslvs.compiler.loader'
    var_4 = 'pyslvs.compiler._site_path'
    var_5 = 'pyslvs.compiler._write'
    var_6 = 'pyslvs.compiler.mkdir'
    var_7 = 'pyslvs.compiler.isdir'
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = {var_8: var_9}

def test_case_0():
    var_0 = 'Test that gen_api uses correct heading level.'
    var_1 = 'pyslvs.compiler.logger'
    var_2 = 'pyslvs.compiler.loader'
    var_3 = 'pyslvs.compiler._site_path'
    var_4 = 'pyslvs.compiler._write'
    var_5 = 'pyslvs.compiler.isdir'
    var_6 = 'MyAPI'
    var_7 = 'myapi'
    var_8 = {var_6: var_7}
    var_9 = 3



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = 'test_pkg'
    var_3 = '__init__.py'
    var_4 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass'
    var_5 = 'Test Package'
    var_6 = {var_5: var_2}
    var_7 = True
    var_8 = False
    var_9 = 'Test Package API'
    var_10 = var_9 in var_1
    var_11 = {var_5: var_10}
    var_12 = 'Non Existent'
    var_13 = 'nonexistent_pkg_xyz'
    var_14 = {var_12: var_13}
    var_15 = 'another_pkg'
    var_16 = '"""Another package."""'
    var_17 = 'Another Package'
    var_18 = {var_5: var_10, var_17: var_15}
    var_19 = 2
    var_20 = 'custom_docs'
    var_21 = {var_5: var_10}
    var_22 = {var_5: var_10}



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = '## Sample Class\n\nThis is a sample class.'
    var_3 = 'Test Module'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'Empty Module'
    var_8 = 'empty_module'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = '## Module 1'
    var_12 = '## Module 2'
    var_13 = 'Module One'
    var_14 = 'Module Two'
    var_15 = 'mod_one'
    var_16 = 'mod_two'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 2
    var_19 = True
    var_20 = 'packages'
    var_21 = str(var_12)
    var_22 = 'Test'
    var_23 = 'test'
    var_24 = {var_22: var_23}
    var_25 = True
    var_26 = module_0.gen_api(var_24, var_21, prefix=var_15, dry=var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = 'My Module'
    var_29 = 'my_module'
    var_30 = {var_28: var_29}
    var_31 = False
    var_32 = module_0.gen_api(var_30, prefix=var_22, dry=var_31)
    var_33 = 'API'
    var_34 = 'api'
    var_35 = {var_33: var_34}
    var_36 = 3
    var_37 = True
    var_38 = module_0.gen_api(var_35, prefix=var_22, level=var_36, dry=var_37)
    var_39 = 'Test'
    var_40 = 'test'
    var_41 = {var_39: var_40}
    var_42 = True
    var_43 = module_0.gen_api(var_41, prefix=var_22, dry=var_42)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_package'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\nVERSION = "1.0.0"'
    var_4 = 'test_module.py'
    var_5 = '"""Test module.\n\nThis is a test module.\n"""\n\ndef test_function():\n    """Test function."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '"""Stub module."""\n\ndef stub_function() -> None: ...\n'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_package'
    var_2 = '__init__.py'
    var_3 = '"""Empty package."""'
    var_4 = False
    var_5 = 2
    var_6 = True

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_test'
    var_2 = '__init__.py'
    var_3 = '"""Package for level test."""'
    var_4 = 'level_test'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'toc_test'
    var_2 = '__init__.py'
    var_3 = '"""Package for TOC test."""'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader without link generation.'
    var_1 = 'nolink_test'
    var_2 = '__init__.py'
    var_3 = '"""Package without links."""'
    var_4 = False
    var_5 = 1



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.submodule'
    var_5 = '/path/to/test_module/submodule'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = "def test_func():\n    '''Test function.'''\n    pass"
    var_9 = '# Compiled API\n'
    var_10 = 'test_root'
    var_11 = '/test/pwd'
    var_12 = True
    var_13 = False
    var_14 = module_0.loader(var_10, var_11, var_12, var_12, var_13)
    assert var_14 == '# Compiled API\n'
    var_15 = True
    var_16 = False

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with extension modules.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '# Compiled\n'
    var_6 = False
    var_7 = True
    var_8 = 'test_root'
    var_9 = '/test/pwd'
    var_10 = 2
    var_11 = module_0.loader(var_8, var_9, var_6, var_10, var_7)
    assert var_11 == '# Compiled\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when no packages are found.'
    var_1 = ''
    var_2 = 'empty_root'
    var_3 = '/test/pwd'
    var_4 = True
    var_5 = False
    var_6 = module_0.loader(var_2, var_3, var_4, var_4, var_5)
    assert var_6 == ''



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'Test Module'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = '/custom/path'
    var_8 = True
    var_9 = {}
    var_10 = True
    var_11 = False
    var_12 = 'Module A'
    var_13 = 'Module B'
    var_14 = 'module_a'
    var_15 = 'module_b'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = True
    var_18 = 2
    var_19 = True
    var_20 = True



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = '\n"""Module docstring."""\n\ndef func():\n    """Function docstring."""\n    pass\n\nclass MyClass:\n    """Class docstring."""\n    pass\n'
    var_6 = 'stub_module.pyi'
    var_7 = '\n"""Stub module docstring."""\n\ndef stub_func() -> None: ...\n'
    var_8 = 'PYTHONPATH'
    var_9 = False
    var_10 = 1

def test_case_0():
    var_0 = 'Test loader function with different options.'
    var_1 = 'test_pkg2'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '\n"""Test module."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_6 = 'PYTHONPATH'
    var_7 = True
    var_8 = 2
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = False
    var_3 = 1

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = False
    var_5 = 1



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = 'packages'
    var_3 = '## Sample Function\n\nThis is a sample function.'
    var_4 = 'Test Module'
    var_5 = 'test_module'
    var_6 = {var_4: var_5}
    var_7 = True
    assert var_7 == 1
    var_8 = False
    var_9 = 'Another Module'
    var_10 = 'another_module'
    var_11 = {var_9: var_10}
    var_12 = True
    assert var_12 == 0
    var_13 = 'Empty Module'
    var_14 = 'empty_module'
    var_15 = {var_13: var_14}
    var_16 = 'Module One'
    var_17 = 'Module Two'
    var_18 = 'module_one'
    var_19 = 'module_two'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 2
    var_22 = module_0.gen_api(var_20, prefix=var_7, level=var_21)
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'test_packages'
    var_2 = 'docs'
    var_3 = '## Module\n\nSample documentation'
    var_4 = 'Test Module'
    var_5 = 'test_module'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = True
    var_9 = False
    var_10 = module_0.gen_api(var_6, var_7, prefix=var_2, link=var_8, level=var_8, toc=var_9, dry=var_8)
    var_11 = len(var_10)
    assert var_11 == 1

def test_case_0():
    var_0 = 'Test gen_api with multiple modules.'
    var_1 = '## Module 1'
    var_2 = '## Module 2'
    var_3 = 'Module One'
    var_4 = 'Module Two'
    var_5 = 'mod1'
    var_6 = 'mod2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'docs'
    var_9 = False
    var_10 = 2
    var_11 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'Empty Module'
    var_2 = 'empty_mod'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'Test'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'Test'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = True

def test_case_0():
    var_0 = 'Test gen_api generates correct file names.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = 'test_module.submodule'
    var_5 = '/path/to/test_module/submodule'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass'
    var_9 = '"""Test stub file."""\ndef test_func() -> None: ...'
    var_10 = True
    var_11 = False
    var_12 = 'test_module'
    var_13 = '/path/to'
    var_14 = True
    var_15 = False
    var_16 = module_0.loader(var_12, var_13, var_14, var_14, var_15)
    assert var_16 == '# Generated Documentation'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with extension modules.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '"""Test module."""'
    var_6 = True
    var_7 = False
    var_8 = 'test_module'
    var_9 = '/path/to'
    var_10 = False
    var_11 = 2
    var_12 = True
    var_13 = module_0.loader(var_8, var_9, var_10, var_11, var_12)
    assert var_13 == '# Documentation'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with empty package.'
    var_1 = 'empty_module'
    var_2 = '/path/to'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == ''



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'docs'
    var_2 = 'Test API'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = '## Function\n\nTest function documentation.'
    var_6 = '/fake/path'
    var_7 = [var_6]
    var_8 = None
    var_9 = True
    var_10 = False

def test_case_0():
    var_0 = 'Test gen_api function with dry run enabled.'
    var_1 = 'docs'
    var_2 = 'Test API'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = '## Function\n\nTest function documentation.'
    var_6 = '/fake/path'
    var_7 = [var_6]
    var_8 = None
    var_9 = True
    var_10 = False

def test_case_0():
    var_0 = 'Test gen_api function when documentation is empty.'
    var_1 = 'docs'
    var_2 = 'Test API'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = '/fake/path'
    var_6 = [var_5]
    var_7 = None
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test gen_api function with custom pwd.'
    var_1 = 'docs'
    var_2 = 'site-packages'
    var_3 = 'Test API'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = '## Function\n\nTest.'
    var_7 = '/fake/path'
    var_8 = [var_7]
    var_9 = True
    var_10 = 2
    var_11 = False



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'test_pkg'
    var_2 = 'docs'
    var_3 = '## Sample Module\n\nThis is a test module.'
    var_4 = 'Test Package'
    var_5 = 'test_pkg'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'Test gen_api function with dry run mode.'
    var_1 = '## Test Module\n\nDocumentation.'
    var_2 = 'My API'
    var_3 = 'my_module'
    var_4 = {var_2: var_3}
    var_5 = 'docs'
    var_6 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'Empty'
    var_2 = 'empty_pkg'
    var_3 = {var_1: var_2}
    var_4 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = '## Module A'
    var_2 = '## Module B'
    var_3 = 'Package A'
    var_4 = 'Package B'
    var_5 = 'pkg_a'
    var_6 = 'pkg_b'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with different heading levels.'
    var_1 = 'API'
    var_2 = 'myapi'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = 3

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = 'Test'
    var_2 = 'test_module_name'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = 0



