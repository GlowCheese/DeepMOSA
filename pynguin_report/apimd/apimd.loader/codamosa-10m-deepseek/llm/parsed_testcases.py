####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'Compiled content'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'mock content'



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_directory'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'Parsed test_package.module1\nParsed test_package.module2\n'



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Example'
    var_1 = 'example'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = len(var_2)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\ndef func():\n    """Test function."""\n    pass'
    var_5 = 'docs'
    var_6 = 'Test Package'
    var_7 = {var_6: var_0}
    var_8 = True
    var_9 = {var_6: var_0}
    var_10 = 'test-pkg-api.md'
    var_11 = 'Non-existent'
    var_12 = 'nonexistent_pkg'
    var_13 = {var_11: var_12}
    var_14 = 'nonexistent-pkg-api.md'
    var_15 = 'Builtin'
    var_16 = 'os'
    var_17 = {var_15: var_16}
    var_18 = 'os-api.md'



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = './test_dir'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_path'
    var_4 = '/tmp/docs'
    var_5 = True
    var_6 = '/tmp/docs/test-package-api.md'
    var_7 = '/tmp/test_path'
    var_8 = '/tmp/test_site_packages'
    var_9 = None



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = 'def test_func():\n    """Test function docstring."""\n    pass\n'
    var_5 = True
    assert var_5 == ''
    var_6 = 2
    var_7 = False
    var_8 = 'nonexistent'
    var_9 = False
    var_10 = 1
    var_11 = True



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n'
    var_5 = 'def test_func():\n'
    var_6 = '    """Test function docstring."""\n'
    var_7 = '    pass\n'
    var_8 = True
    var_9 = False
    var_10 = 'nonexistent'
    var_11 = True
    var_12 = False



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with different scenarios.'
    var_1 = 'test_package'
    var_2 = '/path/to'
    var_3 = True
    var_4 = module_0.loader(var_1, var_2, var_3, var_3, var_3)
    assert var_4 == 'compiled_output'
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_3)
    assert var_5 == 'compiled_output'



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mock file'
    var_1 = 'mock_root'
    var_2 = 'mock_pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == 'Mocked documentation'



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'valid_root'
    var_1 = 'valid_pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = len(var_3)
    var_5 = 'invalid_root'
    var_6 = module_0.loader(var_5, var_1, var_2, var_2, var_2)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 'invalid_pwd'
    var_9 = module_0.loader(var_0, var_8, var_2, var_2, var_2)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = False
    var_12 = module_0.loader(var_0, var_1, var_11, var_2, var_2)
    var_13 = len(var_12)
    var_14 = 2
    var_15 = module_0.loader(var_0, var_1, var_2, var_14, var_2)
    var_16 = len(var_15)
    var_17 = module_0.loader(var_0, var_1, var_2, var_2, var_11)
    var_18 = len(var_17)



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'Parsed test_module'



# Parsed testcases at query #19
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_dir'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = True
    var_6 = '__init__.py'
    var_7 = '"""Test package."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_8 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = '__init__.py'
    var_6 = 'def example_func():\n    pass'
    var_7 = 'module.py'
    var_8 = 'def another_func():\n    pass'
    var_9 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = 'def foo():\n    """Test function."""\n    pass'
    var_3 = '__init__.py'
    var_4 = 'module.py'
    var_5 = False
    var_6 = 1
    var_7 = module_0.loader(var_0, var_1, var_5, var_6, var_5)
    var_8 = True
    var_9 = 2
    var_10 = True
    var_11 = module_0.loader(var_0, var_1, var_8, var_9, var_10)
    var_12 = 'nonexistent'
    var_13 = module_0.loader(var_12, var_1, var_5, var_10, var_5)



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = 0



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\n    \ndef test_func():\n    """Test function docstring."""\n    pass\n    \nclass TestClass:\n    """Test class docstring."""\n    \n    def test_method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = module_0.loader(var_3, var_0, var_4, var_4, var_4)



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/fake/path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'def foo():\n    pass'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_6 == 'module_docstring'



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'ExampleModule'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = True
    var_12 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_11)
    var_13 = None
    var_14 = module_0.gen_api(var_2, var_13, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_11)
    var_15 = {}
    var_16 = module_0.gen_api(var_15, var_13, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_11)
    var_17 = len(var_16)
    assert var_17 == 0



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = True
    var_6 = '__init__.py'
    var_7 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_8 = module_0.loader(var_1, var_0, var_2, var_3, var_4)
    var_9 = 'nonexistent'
    var_10 = 'nonexistent_path'
    var_11 = module_0.loader(var_9, var_10, var_2, var_3, var_4)
    var_12 = 'sys'
    var_13 = ''
    var_14 = module_0.loader(var_12, var_13, var_2, var_3, var_4)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n'
    var_5 = 'def func():\n    """Test function."""\n    pass\n'
    var_6 = True
    assert var_6 == ''
    var_7 = False
    var_8 = 'nonexistent'
    var_9 = '/nonexistent/path'
    var_10 = True
    var_11 = False
    assert var_11 == ''
    var_12 = module_0.loader(var_8, var_9, var_10, var_10, var_11)
    var_13 = 'empty_pkg'
    var_14 = True
    var_15 = False



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'sample.py'
    var_1 = "def sample_func():\n    '''Sample docstring'''\n    pass"
    var_2 = 'sample.pyi'
    var_3 = "def sample_func():\n    '''Sample docstring'''\n    pass"
    var_4 = 'sample'
    var_5 = True
    var_6 = 'nonexistent'
    var_7 = 'sample.so'
    var_8 = 'mock extension module'
    var_9 = '.so'
    var_10 = 'sample'
    var_11 = True



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/fake/path'
    var_4 = 'fake_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = False
    var_11 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_10)
    var_12 = None
    var_13 = True
    var_14 = module_0.gen_api(var_2, var_12, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_13)
    var_15 = 'invalid'
    var_16 = 'invalid_package'
    var_17 = {var_15: var_16}
    var_18 = module_0.gen_api(var_17, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_13)
    var_19 = len(var_18)
    assert var_19 == 0



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = '/mocked/path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == 'compiled_document'



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = '/path/to/pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    assert var_3 == 'Compiled Output'



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'Compiled content'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_6 == 'Compiled content'
    var_7 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_7 == 'Compiled content'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_package'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'invalid_package'
    var_7 = '/path/to/package'
    var_8 = True
    var_9 = 1
    var_10 = True
    var_11 = module_0.loader(var_6, var_7, var_8, var_9, var_10)
    var_12 = 'example_package'
    var_13 = '/invalid/path'
    var_14 = True
    var_15 = 1
    var_16 = True
    var_17 = module_0.loader(var_12, var_13, var_14, var_15, var_16)
    var_18 = 'example_package'
    var_19 = '/path/to/package'
    var_20 = False
    var_21 = 1
    var_22 = True
    var_23 = module_0.loader(var_18, var_19, var_20, var_21, var_22)
    var_24 = 'example_package'
    var_25 = '/path/to/package'
    var_26 = True
    var_27 = 2
    var_28 = True
    var_29 = module_0.loader(var_24, var_25, var_26, var_27, var_28)
    var_30 = 'example_package'
    var_31 = '/path/to/package'
    var_32 = True
    var_33 = 1
    var_34 = False
    var_35 = module_0.loader(var_30, var_31, var_32, var_33, var_34)



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'some_directory'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = None
    var_12 = module_0.gen_api(var_2, var_11, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'non_existent'
    var_15 = 'non_existent_module'
    var_16 = {var_14: var_15}
    var_17 = module_0.gen_api(var_16, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_18 = len(var_17)
    assert var_18 == 0



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = '/path/to/pwd'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'module1: Content of /path/to/module1.py\nmodule2: Content of /path/to/module2.py'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n'
    var_5 = True
    var_6 = False
    var_7 = 'nonexistent'
    var_8 = True
    var_9 = False
    var_10 = 'ext_pkg'
    var_11 = 'ext.so'
    var_12 = 'ext_pkg'
    var_13 = True
    var_14 = False
    var_15 = 'empty_pkg'
    var_16 = True
    var_17 = False



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    assert var_3 == 'Parsed test_module\nLoaded docstring for test_module\n'



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = 'NonExistent'
    var_12 = 'non_existent'
    var_13 = {var_11: var_12}
    var_14 = module_0.gen_api(var_13, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = {var_0: var_1}
    var_17 = None
    var_18 = module_0.gen_api(var_16, var_17, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_19 = len(var_18)
    var_20 = {var_0: var_1}
    var_21 = False
    var_22 = module_0.gen_api(var_20, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_21)
    var_23 = len(var_22)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def example_function():\n    pass\n'
    var_2 = 'test_module.pyi'
    var_3 = 'def example_function():\n    pass\n'
    var_4 = 'test_module.so'
    var_5 = 'Mock extension module content'
    var_6 = 'test_module'
    var_7 = True
    var_8 = 'test_module <= test_module.py'
    var_9 = 'test_module <= test_module.pyi'
    var_10 = 'loading extension module for fully documented:'
    var_11 = 'test_module <= test_module.so'
    var_12 = 'test_module'
    var_13 = 'def example_function():\n    pass\n'



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = 'test_module'
    var_2 = '__init__.py'
    var_3 = '"""Test module docstring."""\n    \ndef test_func():\n    """Test function docstring."""\n    pass\n    \nclass TestClass:\n    """Test class docstring."""\n    \n    def test_method(self):\n        """Test method docstring."""\n        pass\n'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_1, var_0, var_4, var_5, var_4)
    var_7 = True
    var_8 = 2
    var_9 = True
    var_10 = module_0.loader(var_1, var_0, var_7, var_8, var_9)



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mock_root'
    var_1 = '/mock/pwd'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == 'mock_module: mock module docstring'



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_package'
    var_1 = '/path/to/example_package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'compiled_docstring'



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_directory'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'compiled_content'



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = module_0.gen_api(var_1, dry=var_2)
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = '# Test Pkg API'
    var_8 = 'nonexistent'
    var_9 = 'nonexistent_pkg'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_2)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = {var_0: var_0}
    var_14 = 'custom_docs'
    var_15 = module_0.gen_api(var_13, prefix=var_14, dry=var_2)
    var_16 = len(var_15)
    var_17 = False
    var_18 = module_0.gen_api(var_13, link=var_17, dry=var_2)
    var_19 = len(var_18)
    var_20 = 2
    var_21 = module_0.gen_api(var_13, level=var_20, dry=var_2)
    var_22 = var_21[var_17]
    var_23 = '## Test Pkg API'
    var_24 = module_0.gen_api(var_13, toc=var_2, dry=var_2)
    var_25 = len(var_24)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test_package'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = True
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = module_1.isdir(var_4)



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/fake/path'
    var_4 = 'fake_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Example API'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function.'
    var_1 = 'test_pkg'
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_4 = 'test_module.pyi'
    var_5 = '"""Test stub docstring."""\ndef test_func(): ...\n'
    var_6 = True
    var_7 = False
    var_8 = 'nonexistent'

def test_case_0():
    var_0 = 'Test loader with extension modules.'
    var_1 = 'sys.path'
    var_2 = 'test_ext_pkg'
    var_3 = 'test_ext.so'
    var_4 = 'dummy content'
    var_5 = '_load_module'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with invalid/non-Python files.'
    var_1 = 'test_invalid'
    var_2 = 'test.txt'
    var_3 = 'This is not a Python file'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with empty directory.'
    var_1 = 'empty'
    var_2 = True
    var_3 = False



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '/path/to/test_package'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = True
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the loader function with various scenarios.'
    var_1 = 'test_module.py'
    var_2 = '"""Module docstring."""\n\ndef foo():\n    """Function docstring."""\n    pass'
    var_3 = 'test_module.pyi'
    var_4 = '"""Stub docstring."""\n\ndef foo():\n    """Stub function docstring."""\n    ...'
    var_5 = True
    var_6 = False
    var_7 = 'nonexistent'
    var_8 = 'test_module.so'
    var_9 = 2



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_pkg'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = len(var_2)



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Example'
    var_1 = 'example'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = 0



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'docs'
    var_3 = True
    var_4 = False
    var_5 = 'test-module-api.md'
    var_6 = 'non_existent'
    var_7 = {var_6: var_6}
    var_8 = 2



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mock_root'
    var_1 = 'mock_pwd'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = 'Mock Compiled Output'
    var_6 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = {var_0: var_1}
    var_6 = 'test_docs'
    var_7 = module_0.gen_api(var_5, prefix=var_6)
    var_8 = 'nonexistent'
    var_9 = 'nonexistent_module'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = {var_0: var_1}
    var_14 = '/tmp'
    var_15 = module_0.gen_api(var_13, var_14, dry=var_3)
    var_16 = {var_0: var_1}
    var_17 = 2
    var_18 = module_0.gen_api(var_16, level=var_17, dry=var_3)
    var_19 = {var_0: var_1}
    var_20 = False
    var_21 = module_0.gen_api(var_19, toc=var_20, dry=var_3)
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, link=var_20, dry=var_3)



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = '/current/path'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    assert var_3 == 'Compiled content'



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = {var_0: var_1}
    var_6 = 'test_docs'
    var_7 = False
    var_8 = module_0.gen_api(var_5, prefix=var_6, dry=var_7)
    var_9 = 'NonExistent'
    var_10 = 'nonexistent_module'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, dry=var_3)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = '/tmp'
    var_16 = module_0.gen_api(var_14, var_15, dry=var_3)
    var_17 = {var_0: var_1}
    var_18 = 2
    var_19 = module_0.gen_api(var_17, level=var_18, dry=var_3)
    var_20 = {var_0: var_1}
    var_21 = module_0.gen_api(var_20, toc=var_7, dry=var_3)
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, link=var_7, dry=var_3)



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = len(var_2)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test gen_api function.'
    var_1 = 'test_pkg'
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_4 = 'test_module.pyi'
    var_5 = '"""Test module stub docstring."""\ndef test_func() -> None:\n    """Test function stub docstring."""\n    ...\n'
    var_6 = '_site_path'
    var_7 = 'Test Module'
    var_8 = 'test_module'
    var_9 = {var_7: var_8}
    var_10 = 'docs'
    var_11 = True
    var_12 = {var_7: var_8}
    var_13 = 'test-module-api.md'
    var_14 = 'Missing'
    var_15 = 'missing_module'
    var_16 = {var_14: var_15}
    var_17 = {var_7: var_8}
    var_18 = False
    var_19 = {var_7: var_8}
    var_20 = 2
    var_21 = {var_7: var_8}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = {var_0: var_0}
    var_2 = 'test_pkg'
    var_3 = '"""Test package docstring."""\n'
    var_4 = '"""Test module docstring."""\n'
    var_5 = 'docs'
    var_6 = True
    var_7 = False
    var_8 = 'test-pkg-api.md'
    var_9 = 'nonexistent'
    var_10 = 'nonexistent_pkg'
    var_11 = {var_9: var_10}



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_docs'
    var_6 = {var_0: var_1}
    var_7 = module_0.gen_api(var_6, prefix=var_5)
    var_8 = 'Nonexistent'
    var_9 = 'nonexistent_package'
    var_10 = {var_8: var_9}
    var_11 = module_0.gen_api(var_10, dry=var_3)
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = {var_0: var_1}
    var_14 = '/tmp'
    var_15 = module_0.gen_api(var_13, var_14, dry=var_3)
    var_16 = {var_0: var_1}
    var_17 = False
    var_18 = 2
    var_19 = module_0.gen_api(var_16, link=var_17, level=var_18, toc=var_3, dry=var_3)



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test_module'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = len(var_2)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/project'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = 'test_file'
    var_10 = 'r'
    var_11 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_12 = len(var_11)
    assert var_12 == 1



# Parsed testcases at query #20
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_dir'
    var_4 = ''
    var_5 = lambda name: var_3 if name == var_1 else var_4
    var_6 = 'test_package.module1'
    var_7 = 'test_dir/test_package/module1.py'
    var_8 = (var_6, var_7)
    var_9 = 'test_package.module2'
    var_10 = 'test_dir/test_package/module2.py'
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = lambda name, path: var_12
    var_14 = '.py'
    var_15 = "'''Test docstring'''"
    var_16 = lambda path: var_15 if path.endswith(var_14) else var_4
    var_17 = {}
    var_18 = lambda path, doc: written_files.update({path: doc})
    var_19 = 'docs'
    var_20 = False
    var_21 = True
    var_22 = lambda path: var_20 if path == var_19 else var_21
    var_23 = False
    var_24 = module_0.gen_api(var_2, dry=var_21)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = module_0.gen_api(var_2, dry=var_20)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_directory'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = 'non_existent_package'
    var_8 = 'test_directory'
    var_9 = True
    var_10 = 1
    var_11 = True
    var_12 = module_0.loader(var_7, var_8, var_9, var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = 'stub_only_package'
    var_15 = 'test_directory'
    var_16 = True
    var_17 = 1
    var_18 = True
    var_19 = module_0.loader(var_14, var_15, var_16, var_17, var_18)
    var_20 = len(var_19)
    var_21 = 'extension_module_package'
    var_22 = 'test_directory'
    var_23 = True
    var_24 = 1
    var_25 = True
    var_26 = module_0.loader(var_21, var_22, var_23, var_24, var_25)
    var_27 = len(var_26)
    var_28 = 'no_documentation_package'
    var_29 = 'test_directory'
    var_30 = True
    var_31 = 1
    var_32 = True
    var_33 = module_0.loader(var_28, var_29, var_30, var_31, var_32)
    var_34 = len(var_33)
    assert var_34 == 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'mock_pkg'
    var_1 = '__init__.py'
    var_2 = '# Mock package\n'
    var_3 = 'mock_module.py'
    var_4 = '# Mock module\n'
    var_5 = 'mock_module.pyi'
    var_6 = '# Mock stub\n'
    var_7 = 'mock_pkg.mock_module'
    var_8 = 'mock_module'
    var_9 = 1
    var_10 = 'non_existent_pkg'



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = {var_0: var_1}
    var_6 = 'test_docs'
    var_7 = False
    var_8 = module_0.gen_api(var_5, prefix=var_6, dry=var_7)
    var_9 = 'NonExistent'
    var_10 = 'nonexistent_module'
    var_11 = {var_9: var_10}
    var_12 = module_0.gen_api(var_11, dry=var_3)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = {var_0: var_1}
    var_15 = '/custom/path'
    var_16 = module_0.gen_api(var_14, var_15, dry=var_3)
    var_17 = {var_0: var_1}
    var_18 = 2
    var_19 = module_0.gen_api(var_17, level=var_18, dry=var_3)
    var_20 = {var_0: var_1}
    var_21 = module_0.gen_api(var_20, toc=var_7, dry=var_3)
    var_22 = {var_0: var_1}
    var_23 = module_0.gen_api(var_22, link=var_7, dry=var_3)



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'
    var_2 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = False
    var_5 = 'nonexistent_module'
    var_6 = '/nonexistent/path'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    var_10 = 'test_ext'
    var_11 = '__init__.so'
    var_12 = 'test_ext'
    var_13 = True
    var_14 = False
    var_15 = 'test_stub'
    var_16 = '__init__.pyi'
    var_17 = '"""Stub module docstring."""\n\ndef stub_func() -> None:\n    """Stub function docstring."""\n    ...\n'
    var_18 = True
    var_19 = False



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/test/path'
    var_4 = 'docs'
    var_5 = module_0.gen_api(var_2, var_3, prefix=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'test-module-api.md'
    var_8 = 'w'
    var_9 = 'utf-8'
    var_10 = True
    var_11 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = module_0.gen_api(var_2, var_3, prefix=var_4)
    var_14 = len(var_13)
    assert var_14 == 0
    var_15 = module_0.gen_api(var_2, var_3, prefix=var_4)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = None
    var_18 = module_0.gen_api(var_2, var_17, prefix=var_4)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'Module1'
    var_21 = 'Module2'
    var_22 = 'module1'
    var_23 = 'module2'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.gen_api(var_24, var_3, prefix=var_4)
    var_26 = len(var_25)
    assert var_26 == 2



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '.'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = '"""Test package."""'
    var_6 = '"""Test module."""\ndef func():\n    """Test function."""\n    pass'
    var_7 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_package'
    var_1 = 'example_path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = ''
    var_7 = ''
    var_8 = module_0.loader(var_6, var_7, var_2, var_3, var_4)
    var_9 = 'non_existent_package'
    var_10 = 'non_existent_path'
    var_11 = module_0.loader(var_9, var_10, var_2, var_3, var_4)
    var_12 = 2
    var_13 = False
    var_14 = module_0.loader(var_9, var_10, var_2, var_12, var_13)
    var_15 = False
    var_16 = module_0.loader(var_9, var_10, var_15, var_12, var_13)



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda *args, **kwargs: var_0
    var_2 = 'Mock Title'
    var_3 = 'mock_package'
    var_4 = {var_2: var_3}
    var_5 = '/mock/pwd'
    var_6 = 'mock_prefix'
    var_7 = True
    var_8 = False
    var_9 = module_0.gen_api(var_4, var_5, prefix=var_6, link=var_7, level=var_7, toc=var_8, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'valid_root'
    var_1 = 'valid_pwd'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'Expected a string output'
    var_7 = 'non_existent_root'
    var_8 = 'valid_pwd'
    var_9 = module_0.loader(var_7, var_8, var_2, var_3, var_4)
    assert var_9 == ''
    var_10 = 'valid_root'
    var_11 = 'non_existent_pwd'
    var_12 = module_0.loader(var_10, var_11, var_2, var_3, var_4)
    assert var_12 == ''
    var_13 = 'valid_root'
    var_14 = 'valid_pwd'
    var_15 = False
    var_16 = module_0.loader(var_13, var_14, var_15, var_3, var_4)
    var_17 = 'Expected a string output with link set to False'
    var_18 = 'valid_root'
    var_19 = 'valid_pwd'
    var_20 = True
    var_21 = 2
    var_22 = module_0.loader(var_18, var_19, var_20, var_21, var_4)
    var_23 = 'Expected a string output with level set to 2'
    var_24 = 'valid_root'
    var_25 = 'valid_pwd'
    var_26 = True
    var_27 = 1
    var_28 = True
    var_29 = module_0.loader(var_24, var_25, var_26, var_27, var_28)
    var_30 = 'Expected a string output with toc set to True'



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# TestModule API'
    var_9 = 'TestModule'
    var_10 = 'test_module'
    var_11 = {var_9: var_10}
    var_12 = False
    var_13 = len(var_4)
    assert var_13 == 1
    var_14 = 'test-module-api.md'
    var_15 = '# TestModule API'
    var_16 = 'NonExistent'
    var_17 = 'nonexistent_module'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_12)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = 'Module1'
    var_22 = 'Module2'
    var_23 = 'module1'
    var_24 = 'module2'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.gen_api(var_25, dry=var_12)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_26[var_14]
    var_29 = '# Module1 API'
    var_30 = var_26[var_12]
    var_31 = '# Module2 API'
    var_32 = 'TestModule'
    var_33 = 'test_module'
    var_34 = {var_32: var_33}
    var_35 = True



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n'
    var_5 = True
    assert var_5 == ''
    var_6 = False
    var_7 = 'nonexistent_pkg'
    var_8 = '/nonexistent/path'
    var_9 = True
    var_10 = False
    var_11 = module_0.loader(var_7, var_8, var_9, var_9, var_10)
    var_12 = 'ext_pkg'
    var_13 = '__init__.py'
    var_14 = '"""Extension package."""\n'
    var_15 = 'extmod.so'
    var_16 = 'ext_pkg'
    var_17 = True
    var_18 = False



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_directory'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = 2
    var_6 = True
    var_7 = True
    var_8 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    assert var_3 == 'Compiled Output'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring"""\n'
    var_3 = 'module.py'
    assert var_3 == ''
    var_4 = '"""Test module docstring"""\n'
    var_5 = True
    var_6 = False
    var_7 = 'empty_pkg'
    var_8 = '__init__.py'
    var_9 = True
    var_10 = False
    var_11 = 'nonexistent'
    var_12 = True
    var_13 = False



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\n    \ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = 'test_module'
    var_4 = True
    var_5 = False
    var_6 = module_0.loader(var_3, var_0, var_4, var_4, var_5)



# Parsed testcases at query #21
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/site-packages'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = 'importlib.util'
    var_10 = __import__(var_9)
    var_11 = var_10.find_spec
    var_12 = None
    var_13 = '/path/to/docs/example-module-api.md'
    var_14 = '/path/to/site-packages'
    var_15 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 0
    var_18 = var_15[var_17]
    var_19 = '# Example API\n\n'



# Parsed testcases at query #22
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/fake/path'
    var_4 = 'fake_docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = None
    var_11 = module_0.gen_api(var_2, var_10, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'temp_docs'
    var_4 = True
    var_5 = 1
    var_6 = False
    var_7 = False
    var_8 = None
    var_9 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_12 = len(var_11)
    var_13 = True
    var_14 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_13)
    var_15 = len(var_14)
    var_16 = '/path/to/module'
    var_17 = module_0.gen_api(var_2, var_16, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_18 = len(var_17)
    var_19 = 2
    var_20 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_19, toc=var_6, dry=var_7)
    var_21 = len(var_20)
    var_22 = True
    var_23 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_19, toc=var_22, dry=var_7)
    var_24 = len(var_23)



# Parsed testcases at query #25
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestPackage'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/fake/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)



