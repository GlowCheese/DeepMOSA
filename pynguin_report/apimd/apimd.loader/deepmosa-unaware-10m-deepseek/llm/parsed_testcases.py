####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'docs'
    var_6 = False
    var_7 = 'test-module-api.md'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = True
    var_16 = False
    var_17 = module_0.gen_api(var_12, link=var_16, dry=var_3)
    var_18 = module_0.gen_api(var_12, toc=var_3, dry=var_3)
    var_19 = 2
    var_20 = module_0.gen_api(var_12, level=var_19, dry=var_3)
    var_21 = 'NonExistent'
    var_22 = 'nonexistent_module'
    var_23 = {var_21: var_22}
    var_24 = module_0.gen_api(var_23, dry=var_3)
    var_25 = 'new_docs'
    var_26 = True



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '/test/path/pkg'
    var_1 = 'subpkg'
    var_2 = [var_1]
    var_3 = '__init__.py'
    var_4 = 'module.py'
    var_5 = [var_3, var_4]
    var_6 = (var_0, var_2, var_5)
    var_7 = '/test/path/pkg/subpkg'
    var_8 = []
    var_9 = 'submodule.py'
    var_10 = [var_9]
    var_11 = (var_7, var_8, var_10)
    var_12 = 'pkg'
    var_13 = '/test/path'
    var_14 = module_0.walk_packages(var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = '/test/path/pkg'
    var_18 = []
    var_19 = 'module.py'
    var_20 = 'data.txt'
    var_21 = 'config.ini'
    var_22 = [var_19, var_20, var_21]
    var_23 = (var_17, var_18, var_22)
    var_24 = 'pkg'
    var_25 = '/test/path'
    var_26 = module_0.walk_packages(var_24, var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = '/test/path/pkg'
    var_30 = []
    var_31 = 'module.pyi'
    var_32 = 'stub.pyi'
    var_33 = [var_31, var_32]
    var_34 = (var_29, var_30, var_33)
    var_35 = 'pkg'
    var_36 = '/test/path'
    var_37 = module_0.walk_packages(var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '/test/path'
    var_41 = 'pkg'
    var_42 = 'other'
    var_43 = [var_41, var_42]
    var_44 = []
    var_45 = (var_40, var_43, var_44)
    var_46 = '/test/path/pkg'
    var_47 = []
    var_48 = 'module.py'
    var_49 = [var_48]
    var_50 = (var_46, var_47, var_49)
    var_51 = '/test/path/other'
    var_52 = []
    var_53 = 'other.py'
    var_54 = [var_53]
    var_55 = (var_51, var_52, var_54)
    var_56 = module_0.walk_packages(var_41, var_40)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 2
    var_59 = '/test/path/pkg-stubs'
    var_60 = []
    var_61 = '__init__.pyi'
    var_62 = 'module.pyi'
    var_63 = [var_61, var_62]
    var_64 = (var_59, var_60, var_63)
    var_65 = 'pkg'
    var_66 = '/test/path'
    var_67 = module_0.walk_packages(var_65, var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = 'test'
    var_71 = '/path'
    var_72 = module_0.walk_packages(var_70, var_71)
    var_73 = '/test/path/pkg'
    var_74 = 'sub1'
    var_75 = 'sub2'
    var_76 = [var_74, var_75]
    var_77 = '__init__.py'
    var_78 = 'a.py'
    var_79 = 'b.pyi'
    var_80 = [var_77, var_78, var_79]
    var_81 = (var_73, var_76, var_80)
    var_82 = '/test/path/pkg/sub1'
    var_83 = []
    var_84 = 'c.py'
    var_85 = 'd.txt'
    var_86 = [var_84, var_85]
    var_87 = (var_82, var_83, var_86)
    var_88 = '/test/path/pkg/sub2'
    var_89 = []
    var_90 = 'e.pyi'
    var_91 = [var_90]
    var_92 = (var_88, var_89, var_91)
    var_93 = 'pkg'
    var_94 = '/test/path'
    var_95 = module_0.walk_packages(var_93, var_94)
    var_96 = list(var_95)
    var_97 = len(var_96)
    assert var_97 == 5
    var_98 = 'C:\\test\\path\\pkg'
    var_99 = []
    var_100 = 'module.py'
    var_101 = [var_100]
    var_102 = (var_98, var_99, var_101)
    var_103 = 'pkg'
    var_104 = 'C:\\test\\path'
    var_105 = module_0.walk_packages(var_103, var_104)
    var_106 = list(var_105)
    var_107 = len(var_106)
    assert var_107 == 2
    var_108 = any(var_82)



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = 'empty_package'
    var_9 = 'empty'
    var_10 = True
    var_11 = False
    var_12 = module_0.loader(var_9, var_8, var_10, var_10, var_11)
    assert var_12 == ''
    var_13 = 'nested_package'
    var_14 = 'subpackage'
    var_15 = '\n"""Package init."""\nfrom . import submodule\n'
    var_16 = '\n"""Submodule docstring."""\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_17 = ''
    var_18 = 'nested_package'
    var_19 = '.'
    var_20 = False
    var_21 = 2
    var_22 = True
    var_23 = module_0.loader(var_18, var_19, var_20, var_21, var_22)
    var_24 = 'stub_package'
    var_25 = '\n"""Stub module."""\ndef stub_func() -> None:\n    """Stub function."""\n    ...\n'
    var_26 = 'stub_module'
    var_27 = True
    var_28 = False
    var_29 = module_0.loader(var_26, var_24, var_27, var_27, var_28)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = True
    var_4 = 'test_package'
    var_5 = True
    var_6 = False
    var_7 = 2
    var_8 = 'nonexistent'
    var_9 = 'test_stub_package'
    var_10 = 'test_stub.pyi'
    var_11 = '\n"""Stub module docstring."""\ndef stub_func() -> None:\n    """Stub function docstring."""\n    ...\n'
    var_12 = 'test_stub_package'
    var_13 = True
    var_14 = False
    var_15 = 'test_mixed_package'
    var_16 = 'test_mixed.py'
    var_17 = 'test_mixed.pyi'
    var_18 = '\n"""Python module docstring."""\ndef py_func():\n    """Python function docstring."""\n    pass\n'
    var_19 = '\n"""Stub module docstring."""\ndef py_func() -> None:\n    """Stub function docstring."""\n    ...\n'
    var_20 = 'test_mixed_package'
    var_21 = True
    var_22 = False



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'Nonexistent'
    var_6 = 'nonexistent_package'
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = 'test_prefix'
    var_10 = 'Test'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = {var_10: var_11}
    var_14 = False
    var_15 = 2
    var_16 = True
    var_17 = {var_10: var_11}
    var_18 = None
    var_19 = {var_10: var_11}
    var_20 = 'test_module'
    var_21 = (var_20, var_6)
    var_22 = 'test_module.submodule'
    var_23 = 'submodule'
    var_24 = (var_22, var_9)
    var_25 = True
    var_26 = '__init__.py'
    var_27 = '"""Test module docstring."""\n'
    var_28 = 'TestModule'
    var_29 = {var_28: var_27}
    var_30 = len(var_4)
    assert var_30 == 1



# Parsed testcases at query #6
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_2, link=var_7, dry=var_3)
    var_9 = 2
    var_10 = module_0.gen_api(var_2, level=var_9, dry=var_3)
    var_11 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, dry=var_3)
    var_18 = len(var_17)
    var_19 = len(var_16)
    var_20 = '/tmp'
    var_21 = module_0.gen_api(var_2, var_20, dry=var_3)
    var_22 = 'BadModule'
    var_23 = 'non_existent_module_xyz'
    var_24 = {var_22: var_23}
    var_25 = module_0.gen_api(var_24, dry=var_3)



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the loader function with various scenarios.'
    var_1 = '/test/path'
    var_2 = []
    var_3 = 'module.py'
    var_4 = 'module.pyi'
    var_5 = 'other.txt'
    var_6 = [var_3, var_4, var_5]
    var_7 = (var_1, var_2, var_6)
    var_8 = '.py'
    var_9 = '.pyi'
    var_10 = (var_8, var_9)
    var_11 = 'test_pkg'
    var_12 = '/test/pwd'
    var_13 = True
    var_14 = False
    var_15 = module_0.loader(var_11, var_12, var_13, var_13, var_14)
    assert var_15 == 'Compiled documentation'
    var_16 = '/test/path'
    var_17 = []
    var_18 = 'module.py'
    var_19 = 'module.so'
    var_20 = [var_18, var_19]
    var_21 = (var_16, var_17, var_20)
    var_22 = 'test_pkg'
    var_23 = '/test/pwd'
    var_24 = False
    var_25 = 2
    var_26 = True
    var_27 = module_0.loader(var_22, var_23, var_24, var_25, var_26)
    assert var_27 == 'Extension docs'
    var_28 = '/empty/path'
    var_29 = []
    var_30 = 'other.txt'
    var_31 = 'data.csv'
    var_32 = [var_30, var_31]
    var_33 = (var_28, var_29, var_32)
    var_34 = 'empty_pkg'
    var_35 = '/empty/pwd'
    var_36 = True
    var_37 = False
    var_38 = module_0.loader(var_34, var_35, var_36, var_36, var_37)
    assert var_38 == ''
    var_39 = '/mixed/path'
    var_40 = []
    var_41 = 'module.py'
    var_42 = 'module.pyi'
    var_43 = [var_41, var_42]
    var_44 = (var_39, var_40, var_43)
    var_45 = 'mixed_pkg'
    var_46 = '/mixed/pwd'
    var_47 = True
    var_48 = 3
    var_49 = module_0.loader(var_45, var_46, var_47, var_48, var_47)
    assert var_49 == 'Mixed docs'
    var_50 = '/stub/path'
    var_51 = []
    var_52 = 'module.py'
    var_53 = [var_52]
    var_54 = (var_50, var_51, var_53)
    var_55 = '/stub/path/test_pkg-stubs'
    var_56 = []
    var_57 = 'module.pyi'
    var_58 = [var_57]
    var_59 = (var_55, var_56, var_58)
    var_60 = 'test_pkg'
    var_61 = '/stub/pwd'
    var_62 = False
    var_63 = 1
    var_64 = module_0.loader(var_60, var_61, var_62, var_63, var_62)
    assert var_64 == 'Stub filtered docs'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = True
    var_4 = False
    var_5 = 2
    var_6 = 'non_existent'
    var_7 = 'empty_package'
    var_8 = '\n"""Package docstring."""\n__version__ = "1.0.0"\n'
    var_9 = '__init__.py'
    var_10 = '\n"""Type stub for test module."""\nfrom typing import Any\ndef test_func() -> Any: ...\nclass TestClass:\n    def method(self) -> Any: ...\n'
    var_11 = 'test_module.pyi'



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'NonExistent'
    var_6 = 'nonexistent_module'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7, dry=var_3)
    var_9 = 'Module1'
    var_10 = 'Module2'
    var_11 = 'module1'
    var_12 = 'module2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    var_16 = len(var_13)
    var_17 = 'Test'
    var_18 = 'test'
    var_19 = {var_17: var_18}
    var_20 = 'custom_docs'
    var_21 = module_0.gen_api(var_19, prefix=var_20, dry=var_3)
    var_22 = {var_17: var_18}
    var_23 = False
    var_24 = module_0.gen_api(var_22, link=var_23, dry=var_3)
    var_25 = {var_17: var_18}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, level=var_26, dry=var_3)
    var_28 = {var_17: var_18}
    var_29 = module_0.gen_api(var_28, toc=var_3, dry=var_3)
    var_30 = {var_17: var_18}
    var_31 = '/some/path'
    var_32 = module_0.gen_api(var_30, var_31, dry=var_3)



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'docs'
    var_6 = False
    var_7 = 'test-module-api.md'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module_one'
    var_11 = 'module_two'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 2
    var_16 = module_0.gen_api(var_2, level=var_15, dry=var_3)
    var_17 = 0
    var_18 = var_16[var_17]
    var_19 = '## '
    var_20 = False
    var_21 = module_0.gen_api(var_2, toc=var_20, dry=var_3)
    var_22 = False
    var_23 = module_0.gen_api(var_2, link=var_22, dry=var_3)
    var_24 = '/tmp'
    var_25 = module_0.gen_api(var_2, var_24, dry=var_3)
    var_26 = {}
    var_27 = module_0.gen_api(var_26, dry=var_3)
    var_28 = 'Fake'
    var_29 = 'non_existent_module'
    var_30 = {var_28: var_29}
    var_31 = module_0.gen_api(var_30, dry=var_3)



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'NonExistent'
    var_6 = 'non_existent_module'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7, dry=var_3)
    var_9 = 'Test'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = True
    var_13 = 'Test'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = '/tmp'
    var_17 = module_0.gen_api(var_15, var_16, dry=var_12)
    var_18 = {var_13: var_14}
    var_19 = False
    var_20 = 2
    var_21 = module_0.gen_api(var_18, link=var_19, level=var_20, toc=var_12, dry=var_12)
    var_22 = {}
    var_23 = module_0.gen_api(var_22, dry=var_12)
    var_24 = 'Module1'
    var_25 = 'Module2'
    var_26 = 'module1'
    var_27 = 'module2'
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = module_0.gen_api(var_28, dry=var_12)
    var_30 = len(var_29)
    var_31 = len(var_28)



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = module_0.loader(var_4, var_0, var_6, var_5, var_6)
    var_9 = 2
    var_10 = module_0.loader(var_4, var_0, var_5, var_9, var_6)
    var_11 = module_0.loader(var_4, var_0, var_5, var_5, var_5)
    var_12 = 'nonexistent'
    var_13 = module_0.loader(var_12, var_0, var_5, var_5, var_6)
    var_14 = 'test_package'
    var_15 = '__init__.py'
    var_16 = 'module.py'
    var_17 = '"""Package docstring."""'
    var_18 = '\n"""Module docstring."""\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_19 = 'test_package'
    var_20 = '.'
    var_21 = True
    var_22 = False
    var_23 = module_0.loader(var_19, var_20, var_21, var_21, var_22)
    var_24 = 'test_stub_package'
    var_25 = 'module.pyi'
    var_26 = '\n"""Stub module docstring."""\ndef stub_func() -> None:\n    """Stub function docstring."""\n    ...\n'
    var_27 = 'test_stub_package'
    var_28 = '.'
    var_29 = True
    var_30 = False
    var_31 = module_0.loader(var_27, var_28, var_29, var_29, var_30)



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'AnotherModule'
    var_8 = 'another_module'
    var_9 = {var_7: var_8}
    var_10 = '/tmp'
    var_11 = module_0.gen_api(var_9, var_10, dry=var_4)
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'Module3'
    var_15 = 'module1'
    var_16 = 'module2'
    var_17 = 'module3'
    var_18 = {var_12: var_15, var_13: var_16, var_14: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_4)
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = {var_1: var_2}
    var_22 = 'custom_docs'
    var_23 = module_0.gen_api(var_21, prefix=var_22, dry=var_4)
    var_24 = {var_1: var_2}
    var_25 = False
    var_26 = module_0.gen_api(var_24, link=var_25, dry=var_4)
    var_27 = {var_1: var_2}
    var_28 = 2
    var_29 = module_0.gen_api(var_27, level=var_28, dry=var_4)
    var_30 = {var_1: var_2}
    var_31 = module_0.gen_api(var_30, toc=var_4, dry=var_4)
    var_32 = {}
    var_33 = module_0.gen_api(var_32, dry=var_4)
    var_34 = 'NonExistent'
    var_35 = 'nonexistent_module_xyz'
    var_36 = {var_34: var_35}
    var_37 = module_0.gen_api(var_36, dry=var_4)
    var_38 = 'Test-Module'
    var_39 = 'test_module_with_underscore'
    var_40 = {var_38: var_39}
    var_41 = module_0.gen_api(var_40, dry=var_4)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    \n    def method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = True
    var_4 = False
    var_5 = 2
    var_6 = 'nonexistent'
    var_7 = 'empty_package'
    var_8 = 'empty_package'
    var_9 = True
    var_10 = False



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_package'
    var_5 = '.'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)
    var_9 = 'test_package'
    var_10 = '.'
    var_11 = True
    var_12 = False
    var_13 = module_0.loader(var_9, var_10, var_11, var_11, var_12)
    var_14 = 'test_package'
    var_15 = '.'
    var_16 = False
    var_17 = 2
    var_18 = module_0.loader(var_14, var_15, var_16, var_17, var_16)
    var_19 = 'test_package'
    var_20 = '.'
    var_21 = False
    var_22 = 1
    var_23 = True
    var_24 = module_0.loader(var_19, var_20, var_21, var_22, var_23)
    var_25 = 'empty_package'
    var_26 = 'empty_package'
    var_27 = '.'
    var_28 = False
    var_29 = 1
    var_30 = module_0.loader(var_26, var_27, var_28, var_29, var_28)
    assert var_30 == ''
    var_31 = 'parent_package'
    var_32 = 'subpackage'
    var_33 = '\n"""Parent package."""\n'
    var_34 = '\n"""Subpackage."""\n'
    var_35 = 'parent_package'
    var_36 = '.'
    var_37 = False
    var_38 = 1
    var_39 = module_0.loader(var_35, var_36, var_37, var_38, var_37)



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = '__init__.py'
    var_7 = '"""Test module docstring."""\n'
    var_8 = 'TestModule'
    var_9 = {var_8: var_7}
    var_10 = False
    var_11 = 'test-module-api.md'
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = len(var_4)
    assert var_17 == 2
    var_18 = 'NonExistent'
    var_19 = 'nonexistent_module'
    var_20 = {var_18: var_19}
    var_21 = {var_8: var_7}
    var_22 = {var_8: var_7}
    var_23 = 2
    var_24 = {var_8: var_7}
    var_25 = True
    var_26 = 'existing_prefix'
    var_27 = {}
    var_28 = None



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'new_docs'
    var_6 = module_0.gen_api(var_2, prefix=var_1)
    var_7 = 'existing_docs'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = len(var_6)
    var_14 = len(var_12)
    var_15 = False
    var_16 = 2
    var_17 = True
    var_18 = {}
    var_19 = 'BadModule'
    var_20 = 'non_existent_module'
    var_21 = {var_19: var_20}



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = 'empty_package'
    var_9 = 'empty'
    var_10 = False
    var_11 = 2
    var_12 = True
    var_13 = module_0.loader(var_9, var_8, var_10, var_11, var_12)
    assert var_13 == ''
    var_14 = 'nested_package'
    var_15 = 'subpackage'
    var_16 = '\n"""Package init."""\nfrom . import module\n'
    var_17 = '\n"""Submodule docstring."""\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_18 = '"""Subpackage init."""'
    var_19 = 'nested_package'
    var_20 = '.'
    var_21 = True
    var_22 = False
    var_23 = module_0.loader(var_19, var_20, var_21, var_21, var_22)
    var_24 = 'stub_package'
    var_25 = '\n"""Stub module."""\ndef stub_func() -> None:\n    """Stub function."""\n    ...\n'
    var_26 = 'stub_module'
    var_27 = False
    var_28 = 1
    var_29 = module_0.loader(var_26, var_24, var_27, var_28, var_27)
    var_30 = 'dual_package'
    var_31 = '\n"""Python module."""\ndef py_func():\n    """Python function."""\n    pass\n'
    var_32 = '\n"""Stub module."""\ndef stub_func() -> None:\n    """Stub function."""\n    ...\n'
    var_33 = 'dual_module'
    var_34 = True
    var_35 = False
    var_36 = module_0.loader(var_33, var_30, var_34, var_34, var_35)



# Parsed testcases at query #19
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = module_0.loader(var_4, var_0, var_6, var_5, var_6)
    var_9 = 2
    var_10 = module_0.loader(var_4, var_0, var_5, var_9, var_5)
    var_11 = 'nonexistent'
    var_12 = module_0.loader(var_11, var_0, var_5, var_5, var_6)
    assert var_12 == ''
    var_13 = 'test_package_dir'
    var_14 = 'mypackage'
    var_15 = '__init__.py'
    var_16 = 'module.py'
    var_17 = '"""Package docstring."""\n'
    var_18 = '\n"""Module docstring."""\nclass TestClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_19 = True
    var_20 = False
    var_21 = module_0.loader(var_14, var_13, var_19, var_19, var_20)
    var_22 = 'test_stub_package'
    var_23 = 'stubpackage'
    var_24 = '__init__.pyi'
    var_25 = '\n"""Stub package docstring."""\ndef stub_func() -> None:\n    """Stub function docstring."""\n    ...\n'
    var_26 = True
    var_27 = False
    var_28 = module_0.loader(var_23, var_22, var_26, var_26, var_27)



# Parsed testcases at query #20
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'docs'
    var_6 = False
    var_7 = 'test-module-api.md'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = True
    var_16 = 'Test'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = False
    var_20 = 2
    var_21 = module_0.gen_api(var_18, link=var_19, level=var_20, toc=var_3, dry=var_3)
    var_22 = {}
    var_23 = module_0.gen_api(var_22, dry=var_3)
    var_24 = 'NonExistent'
    var_25 = 'nonexistent_module'
    var_26 = {var_24: var_25}
    var_27 = module_0.gen_api(var_26, dry=var_3)



# Parsed testcases at query #21
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = var_0 / var_5
    var_7 = '__init__.py'
    var_8 = var_6 / var_7
    var_9 = '"""Test module docstring."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_10 = 'submodule.py'
    var_11 = var_6 / var_10
    var_12 = '"""Submodule docstring."""\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_13 = 'docs'
    var_14 = 'TestModule'
    var_15 = {var_14: var_5}
    var_16 = False
    var_17 = 2
    var_18 = True
    var_19 = len(var_4)
    assert var_19 == 1
    var_20 = 'test-module-api.md'
    var_21 = 'NonExistent'
    var_22 = 'nonexistent_module'
    var_23 = {var_21: var_22}
    var_24 = module_0.gen_api(var_23, dry=var_3)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = 'Module1'
    var_27 = 'Module2'
    var_28 = 'module1'
    var_29 = 'module2'
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = module_0.gen_api(var_30, dry=var_3)
    var_32 = len(var_31)
    var_33 = module_0.gen_api(var_30)
    var_34 = {}
    var_35 = module_0.gen_api(var_34)
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = 0
    var_38 = site.getsitepackages()[var_37]
    var_39 = 'SitePackage'
    var_40 = 'os'
    var_41 = {var_39: var_40}
    var_42 = module_0.gen_api(var_41, var_38, dry=var_3)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_5 = False
    var_6 = 1
    var_7 = True
    var_8 = 2
    var_9 = True
    var_10 = 'nonexistent'
    var_11 = 'module.pyi'
    var_12 = '"""Stub module."""\n\nclass StubClass:\n    """Stub class."""\n    pass\n'

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = False
    var_4 = 1
    var_5 = 'stub_pkg'
    var_6 = '__init__.pyi'
    var_7 = '"""Stub package."""\n'
    var_8 = False
    var_9 = 1
    var_10 = 'deep'
    var_11 = 'nested'
    var_12 = 'package'
    var_13 = var_7 / var_12
    var_14 = True
    var_15 = '__init__.py'
    var_16 = var_13 / var_15
    var_17 = '"""Deep package."""\n'
    var_18 = 'deep.nested.package'
    var_19 = False



# Parsed testcases at query #23
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'test_docs'
    var_8 = True
    var_9 = False
    var_10 = module_0.gen_api(var_3, link=var_9, dry=var_4)
    var_11 = 2
    var_12 = module_0.gen_api(var_3, level=var_11, dry=var_4)
    var_13 = module_0.gen_api(var_3, toc=var_4, dry=var_4)
    var_14 = 'Module1'
    var_15 = 'Module2'
    var_16 = 'module1'
    var_17 = 'module2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_4)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = '/tmp'
    var_22 = module_0.gen_api(var_3, var_21, dry=var_4)
    var_23 = 'Ghost'
    var_24 = 'non_existent_module_xyz'
    var_25 = {var_23: var_24}
    var_26 = module_0.gen_api(var_25, dry=var_4)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '"""\nTest package docstring.\n"""\nfrom .module import hello\n\n__version__ = "1.0.0"\n'
    var_2 = 'def hello(name: str) -> str:\n    """\n    Say hello to someone.\n    \n    Args:\n        name: The name to greet.\n    \n    Returns:\n        A greeting message.\n    """\n    return f"Hello {name}!"\n\nclass TestClass:\n    """A test class."""\n    \n    def method(self) -> int:\n        """Return 42."""\n        return 42\n'
    var_3 = True
    var_4 = False
    var_5 = 2
    var_6 = 'nonexistent'
    var_7 = '"""\nType stubs for test package.\n"""\nfrom typing import Protocol\n\nclass ExampleProtocol(Protocol):\n    """An example protocol."""\n    \n    def method(self) -> int: ...\n'
    var_8 = 'module.cpython-39-x86_64-linux-gnu.so'
    var_9 = b'fake binary'
    var_10 = 'test_pkg'
    var_11 = True
    var_12 = False
    var_13 = 'pure_pkg'
    var_14 = 'def pure_func():\n    """A pure Python function."""\n    pass\n'
    var_15 = 'nested_pkg'
    var_16 = 'subpkg'
    var_17 = '"""\nNested package.\n"""\n'
    var_18 = 'def nested_func():\n    """Nested function."""\n    pass\n'



# Parsed testcases at query #25
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
    var_8 = '# TestModule API\n\n'
    var_9 = 'test_module'
    var_10 = '__init__.py'
    var_11 = '"""Test module docstring."""\n'
    var_12 = False
    var_13 = len(var_4)
    assert var_13 == 1
    var_14 = 'test-module-api.md'
    var_15 = '# TestModule API\n\n'
    var_16 = 'NonExistent'
    var_17 = 'nonexistent_module'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_13)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = 'Module1'
    var_22 = 'Module2'
    var_23 = 'module1'
    var_24 = 'module2'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = module_0.gen_api(var_25, dry=var_13)
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = 'custom_prefix'
    var_29 = 'Test'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = False
    var_33 = len(var_26)
    assert var_33 == 0
    var_34 = 'Test'
    var_35 = 'test'
    var_36 = {var_34: var_35}
    var_37 = False
    var_38 = 2
    var_39 = module_0.gen_api(var_36, link=var_37, level=var_38, toc=var_30, dry=var_30)
    var_40 = 0
    var_41 = var_39[var_40]
    var_42 = '## Test API\n\n'
    var_43 = {}
    var_44 = module_0.gen_api(var_43, dry=var_42)
    var_45 = len(var_44)
    assert var_45 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestPackage'
    var_2 = 'test_package'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'test_package'
    var_7 = '"""Test package."""\n'
    var_8 = True
    var_9 = 'docs'
    var_10 = True
    var_11 = {}
    var_12 = module_0.gen_api(var_11, dry=var_4)
    var_13 = 'Package1'
    var_14 = 'Package2'
    var_15 = 'package1'
    var_16 = 'package2'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.gen_api(var_17, dry=var_4)
    var_19 = len(var_18)
    var_20 = False
    var_21 = module_0.gen_api(var_17, link=var_20, dry=var_4)
    var_22 = 2
    var_23 = module_0.gen_api(var_17, level=var_22, dry=var_4)
    var_24 = module_0.gen_api(var_17, toc=var_4, dry=var_4)
    var_25 = 'NonExistent'
    var_26 = 'nonexistent_package_xyz'
    var_27 = {var_25: var_26}
    var_28 = module_0.gen_api(var_27, dry=var_4)
    var_29 = 'Test_Package'
    var_30 = 'test_package_with_underscore'
    var_31 = {var_29: var_30}
    var_32 = module_0.gen_api(var_31, dry=var_4)



# Parsed testcases at query #2
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'NonExistent'
    var_6 = 'non_existent_module'
    var_7 = {var_5: var_6}
    var_8 = module_0.gen_api(var_7, dry=var_3)
    var_9 = 'Module1'
    var_10 = 'Module2'
    var_11 = 'module1'
    var_12 = 'module2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    var_16 = len(var_13)
    var_17 = 'Test'
    var_18 = 'test'
    var_19 = {var_17: var_18}
    var_20 = 'custom_docs'
    var_21 = module_0.gen_api(var_19, prefix=var_20, dry=var_3)
    var_22 = {var_17: var_18}
    var_23 = False
    var_24 = module_0.gen_api(var_22, link=var_23, dry=var_3)
    var_25 = {var_17: var_18}
    var_26 = 2
    var_27 = module_0.gen_api(var_25, level=var_26, dry=var_3)
    var_28 = {var_17: var_18}
    var_29 = module_0.gen_api(var_28, toc=var_3, dry=var_3)
    var_30 = {var_17: var_18}
    var_31 = '/some/path'
    var_32 = module_0.gen_api(var_30, var_31, dry=var_3)
    var_33 = {}
    var_34 = module_0.gen_api(var_33, dry=var_3)
    var_35 = {var_17: var_18}
    var_36 = module_0.gen_api(var_35, dry=var_3)
    var_37 = all(var_0)



# Parsed testcases at query #3
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = False
    var_6 = 1
    var_7 = module_0.loader(var_4, var_0, var_5, var_6, var_5)
    var_8 = True
    var_9 = module_0.loader(var_4, var_0, var_8, var_8, var_5)
    var_10 = 2
    var_11 = module_0.loader(var_4, var_0, var_5, var_10, var_5)
    var_12 = True
    var_13 = module_0.loader(var_4, var_0, var_5, var_8, var_12)
    var_14 = 'nonexistent'
    var_15 = '.'
    var_16 = False
    var_17 = module_0.loader(var_14, var_15, var_16, var_4, var_16)
    assert var_17 == ''
    var_18 = 'empty_package'
    var_19 = 'empty'
    var_20 = False
    var_21 = 1
    var_22 = module_0.loader(var_19, var_18, var_20, var_21, var_20)
    assert var_22 == ''



# Parsed testcases at query #4
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = '/test/path/pkg'
    var_1 = 'subpkg'
    var_2 = [var_1]
    var_3 = '__init__.py'
    var_4 = 'module.py'
    var_5 = [var_3, var_4]
    var_6 = (var_0, var_2, var_5)
    var_7 = '/test/path/pkg/subpkg'
    var_8 = []
    var_9 = 'submodule.py'
    var_10 = [var_9]
    var_11 = (var_7, var_8, var_10)
    var_12 = '/test/path/pkg-stubs'
    var_13 = []
    var_14 = '__init__.pyi'
    var_15 = 'module.pyi'
    var_16 = [var_14, var_15]
    var_17 = (var_12, var_13, var_16)
    var_18 = 'pkg'
    var_19 = '/test/path'
    var_20 = module_0.walk_packages(var_18, var_19)
    var_21 = list(var_20)
    var_22 = (var_18, var_0)
    var_23 = 'pkg.module'
    var_24 = '/test/path/pkg/module'
    var_25 = (var_23, var_24)
    var_26 = 'pkg.subpkg'
    var_27 = (var_26, var_7)
    var_28 = 'pkg.subpkg.submodule'
    var_29 = '/test/path/pkg/subpkg/submodule'
    var_30 = (var_28, var_29)
    var_31 = [var_22, var_25, var_27, var_30]
    var_32 = sorted(var_21)
    var_33 = sorted(var_31)
    var_34 = '/test/path/pkg'
    var_35 = []
    var_36 = 'module.py'
    var_37 = 'data.txt'
    var_38 = 'config.ini'
    var_39 = [var_36, var_37, var_38]
    var_40 = (var_34, var_35, var_39)
    var_41 = 'pkg'
    var_42 = '/test/path'
    var_43 = module_0.walk_packages(var_41, var_42)
    var_44 = list(var_43)
    var_45 = 'pkg.module'
    var_46 = '/test/path/pkg/module'
    var_47 = (var_45, var_46)
    var_48 = [var_47]
    var_49 = '/test/path/pkg-stubs'
    var_50 = []
    var_51 = '__init__.pyi'
    var_52 = 'module.pyi'
    var_53 = [var_51, var_52]
    var_54 = (var_49, var_50, var_53)
    var_55 = 'pkg'
    var_56 = '/test/path'
    var_57 = module_0.walk_packages(var_55, var_56)
    var_58 = list(var_57)
    var_59 = (var_55, var_49)
    var_60 = 'pkg.module'
    var_61 = '/test/path/pkg-stubs/module'
    var_62 = (var_60, var_61)
    var_63 = [var_59, var_62]
    var_64 = sorted(var_58)
    var_65 = sorted(var_63)
    var_66 = '/test/path'
    var_67 = 'pkg'
    var_68 = 'other'
    var_69 = [var_67, var_68]
    var_70 = []
    var_71 = (var_66, var_69, var_70)
    var_72 = '/test/path/pkg'
    var_73 = []
    var_74 = 'module.py'
    var_75 = [var_74]
    var_76 = (var_72, var_73, var_75)
    var_77 = '/test/path/other'
    var_78 = []
    var_79 = [var_74]
    var_80 = (var_77, var_78, var_79)
    var_81 = module_0.walk_packages(var_67, var_66)
    var_82 = list(var_81)
    var_83 = 'pkg.module'
    var_84 = '/test/path/pkg/module'
    var_85 = (var_83, var_84)
    var_86 = [var_85]
    var_87 = '/test/path/pkg'
    var_88 = []
    var_89 = '__init__.py'
    var_90 = [var_89]
    var_91 = (var_87, var_88, var_90)
    var_92 = 'pkg'
    var_93 = '/test/path'
    var_94 = module_0.walk_packages(var_92, var_93)
    var_95 = list(var_94)
    var_96 = (var_92, var_87)
    var_97 = [var_96]
    var_98 = '/test/path/pkg'
    var_99 = []
    var_100 = 'module.py'
    var_101 = 'module.pyi'
    var_102 = [var_100, var_101]
    var_103 = (var_98, var_99, var_102)
    var_104 = 'pkg'
    var_105 = '/test/path'
    var_106 = module_0.walk_packages(var_104, var_105)
    var_107 = list(var_106)
    var_108 = 'pkg.module'
    var_109 = '/test/path/pkg/module'
    var_110 = (var_108, var_109)
    var_111 = [var_110]
    var_112 = 'pkg'
    var_113 = '/test/path'
    var_114 = module_0.walk_packages(var_112, var_113)
    var_115 = list(var_114)
    var_116 = '/test/path/pkg'
    var_117 = []
    var_118 = 'module.py'
    var_119 = [var_118]
    var_120 = (var_116, var_117, var_119)
    var_121 = 'pkg'
    var_122 = '/test/path/'
    var_123 = module_0.walk_packages(var_121, var_122)
    var_124 = list(var_123)
    var_125 = 'pkg.module'
    var_126 = '/test/path/pkg/module'
    var_127 = (var_125, var_126)
    var_128 = [var_127]



# Parsed testcases at query #5
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = len(var_7)
    var_9 = module_0.loader(var_4, var_0, var_6, var_5, var_6)
    var_10 = 2
    var_11 = module_0.loader(var_4, var_0, var_5, var_10, var_6)
    var_12 = module_0.loader(var_4, var_0, var_5, var_5, var_5)
    var_13 = 'non_existent'
    var_14 = module_0.loader(var_13, var_0, var_5, var_5, var_6)
    assert var_14 == ''
    var_15 = 'test_package_pkg'
    var_16 = '__init__.py'
    var_17 = 'submodule.py'
    var_18 = '"""Package docstring."""'
    var_19 = 'def sub_func():\n    """Sub function."""\n    pass'
    var_20 = 'test_package_pkg'
    var_21 = '.'
    var_22 = True
    var_23 = False
    var_24 = module_0.loader(var_20, var_21, var_22, var_22, var_23)
    var_25 = 'test_stub_pkg'
    var_26 = 'module.pyi'
    var_27 = 'def stub_func() -> None:\n    """Stub function."""\n    ...'
    var_28 = 'test_stub_pkg'
    var_29 = '.'
    var_30 = True
    var_31 = False
    var_32 = module_0.loader(var_28, var_29, var_30, var_30, var_31)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_3 = False
    var_4 = 1
    var_5 = True
    var_6 = 2
    var_7 = True
    var_8 = 'nonexistent'
    var_9 = 'empty_package'
    var_10 = 'ext_package'
    var_11 = '\n"""Extension package."""\n# This file exists but doesn\'t have full documentation\n'
    var_12 = '__init__.py'
    var_13 = '\n"""Extension package stub with full docs."""\ndef documented_func() -> None:\n    """A documented function."""\n    ...\nclass DocumentedClass:\n    """A documented class."""\n    ...\n'
    var_14 = '__init__.pyi'
    var_15 = False
    var_16 = 1
    var_17 = 'nested'
    var_18 = '"""Nested package."""'
    var_19 = 'submodule'
    var_20 = '"""Submodule."""'
    var_21 = False
    var_22 = 1



# Parsed testcases at query #7
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = 'empty_package'
    var_9 = 'empty'
    var_10 = False
    var_11 = 2
    var_12 = True
    var_13 = module_0.loader(var_9, var_8, var_10, var_11, var_12)
    assert var_13 == ''
    var_14 = 'nested_package'
    var_15 = 'subpackage'
    var_16 = '\n"""Subpackage init."""\nfrom .module import func\n'
    var_17 = '\n"""Subpackage module."""\ndef func():\n    """A function."""\n    return True\n'
    var_18 = 'subpackage'
    var_19 = True
    var_20 = False
    var_21 = module_0.loader(var_18, var_14, var_19, var_19, var_20)
    var_22 = 'stub_package'
    var_23 = '\n"""Stub module."""\ndef stub_func() -> bool:\n    """Stub function."""\n    ...\n'
    var_24 = 'stub_module'
    var_25 = False
    var_26 = 1
    var_27 = module_0.loader(var_24, var_22, var_25, var_26, var_25)



# Parsed testcases at query #8
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'TempModule'
    var_7 = 'temp_module'
    var_8 = {var_6: var_7}
    var_9 = 'PrefixTest'
    var_10 = 'prefix_test'
    var_11 = {var_9: var_10}
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'Module3'
    var_15 = 'module_one'
    var_16 = 'module_two'
    var_17 = 'module_three'
    var_18 = {var_12: var_15, var_13: var_16, var_14: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_4)
    var_20 = len(var_19)
    var_21 = len(var_18)
    var_22 = 'NoLink'
    var_23 = 'no_link'
    var_24 = {var_22: var_23}
    var_25 = False
    var_26 = module_0.gen_api(var_24, link=var_25, dry=var_4)
    var_27 = 'LevelTest'
    var_28 = 'level_test'
    var_29 = {var_27: var_28}
    var_30 = 2
    var_31 = module_0.gen_api(var_29, level=var_30, dry=var_4)
    var_32 = 'TocTest'
    var_33 = 'toc_test'
    var_34 = {var_32: var_33}
    var_35 = module_0.gen_api(var_34, toc=var_4, dry=var_4)
    var_36 = {}
    var_37 = module_0.gen_api(var_36, dry=var_4)
    var_38 = 'NonExistent'
    var_39 = 'nonexistent_module_xyz'
    var_40 = {var_38: var_39}
    var_41 = module_0.gen_api(var_40, dry=var_4)
    var_42 = 'Underscore_Test'
    var_43 = 'underscore_module_test'
    var_44 = {var_42: var_43}
    var_45 = module_0.gen_api(var_44, dry=var_4)



# Parsed testcases at query #9
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = {}
    var_6 = module_0.gen_api(var_5, dry=var_3)
    var_7 = 'NonExistent'
    var_8 = 'nonexistent_module'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, dry=var_3)
    var_11 = 'Test'
    var_12 = 'test'
    var_13 = {var_11: var_12}
    var_14 = 'custom_docs'
    var_15 = module_0.gen_api(var_13, prefix=var_14, dry=var_3)
    var_16 = {var_11: var_12}
    var_17 = False
    var_18 = module_0.gen_api(var_16, link=var_17, dry=var_3)
    var_19 = 'Test'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = True
    var_23 = {var_11: var_12}
    var_24 = module_0.gen_api(var_23, toc=var_22, dry=var_22)
    var_25 = {var_11: var_12}
    var_26 = '/tmp'
    var_27 = module_0.gen_api(var_25, var_26, dry=var_22)
    var_28 = 'Module1'
    var_29 = 'Module2'
    var_30 = 'module1'
    var_31 = 'module2'
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = module_0.gen_api(var_32, dry=var_22)
    var_34 = len(var_33)
    var_35 = len(var_32)



# Parsed testcases at query #10
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/fake/path/test_module'
    var_2 = (var_0, var_1)
    var_3 = 'Test Title'
    var_4 = 'test_package'
    var_5 = {var_3: var_4}
    var_6 = '/fake/pwd'
    var_7 = 'docs'
    var_8 = True
    var_9 = False
    var_10 = module_0.gen_api(var_5, var_6, prefix=var_7, link=var_8, level=var_8, toc=var_9, dry=var_8)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = var_10[var_9]
    var_13 = '# Test Title API\n'
    var_14 = 'Load root: test_package (Test Title)'
    var_15 = 'Test'
    var_16 = 'package'
    var_17 = {var_15: var_16}
    var_18 = 'new_docs'
    var_19 = module_0.gen_api(var_17, prefix=var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'Empty'
    var_22 = 'empty_package'
    var_23 = {var_21: var_22}
    var_24 = module_0.gen_api(var_23)
    var_25 = "'empty_package' can not be found"
    var_26 = len(var_24)
    assert var_26 == 1
    var_27 = 'pkg1.mod1'
    var_28 = '/path/pkg1/mod1'
    var_29 = (var_27, var_28)
    var_30 = [var_29]
    var_31 = 'pkg2.mod2'
    var_32 = '/path/pkg2/mod2'
    var_33 = (var_31, var_32)
    var_34 = [var_33]
    var_35 = 'First'
    var_36 = 'Second'
    var_37 = 'package1'
    var_38 = 'package2'
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = 2
    var_41 = module_0.gen_api(var_39, level=var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 0
    var_44 = var_41[var_43]
    var_45 = '## First API\n'
    var_46 = 1
    var_47 = var_41[var_46]
    var_48 = '## Second API\n'
    var_49 = 'docs'
    var_50 = 'test_pkg'
    var_51 = '/fake/path/test_pkg'
    var_52 = (var_50, var_51)
    var_53 = 'Test'
    var_54 = {var_53: var_50}
    var_55 = False
    var_56 = 'test-pkg-api.md'
    var_57 = '# Test API\n'
    var_58 = len(var_41)
    assert var_58 == 1
    var_59 = 'Test'
    var_60 = 'package'
    var_61 = {var_59: var_60}
    var_62 = '/custom/path'
    var_63 = module_0.gen_api(var_61, var_62)



# Parsed testcases at query #11
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'test_module'
    var_7 = '"""Test module."""'
    var_8 = True
    var_9 = len(var_5)
    assert var_9 == 1
    var_10 = 'docs'
    var_11 = True
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, dry=var_9)
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = 'Test'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = False
    var_23 = 2
    var_24 = module_0.gen_api(var_21, link=var_22, level=var_23, toc=var_9, dry=var_9)
    var_25 = {}
    var_26 = module_0.gen_api(var_25, dry=var_9)
    var_27 = 'NonExistent'
    var_28 = 'nonexistent_module'
    var_29 = {var_27: var_28}
    var_30 = module_0.gen_api(var_29, dry=var_9)



# Parsed testcases at query #12
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = '__init__.py'
    var_7 = '"""Test module docstring."""\n\n'
    var_8 = 'def test_func():\n'
    var_9 = '    """Test function."""\n'
    var_10 = '    pass\n'
    var_11 = 'docs'
    var_12 = False
    var_13 = 'test-module-api.md'
    var_14 = 'Module1'
    var_15 = 'Module2'
    var_16 = {var_14: var_7, var_15: var_7}
    var_17 = True
    var_18 = 'BadModule'
    var_19 = 'nonexistent_module'
    var_20 = {var_18: var_19}
    var_21 = module_0.gen_api(var_20, dry=var_17)
    var_22 = 2



# Parsed testcases at query #13
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = None
    var_6 = module_0.gen_api(var_2, var_5, dry=var_3)
    var_7 = {}
    var_8 = module_0.gen_api(var_7, dry=var_3)
    var_9 = 'Module1'
    var_10 = 'Module2'
    var_11 = 'module1'
    var_12 = 'module2'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.gen_api(var_13, dry=var_3)
    var_15 = len(var_14)
    var_16 = len(var_13)
    var_17 = 2
    var_18 = module_0.gen_api(var_2, level=var_17, dry=var_3)
    var_19 = False
    var_20 = module_0.gen_api(var_2, toc=var_19, dry=var_3)
    var_21 = module_0.gen_api(var_2, link=var_19, dry=var_3)



# Parsed testcases at query #14
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'docs'
    var_6 = False
    var_7 = 'test-module-api.md'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = len(var_13)
    var_15 = len(var_12)
    var_16 = 2
    var_17 = module_0.gen_api(var_12, level=var_16, dry=var_3)
    var_18 = '## '
    var_19 = False
    var_20 = module_0.gen_api(var_12, toc=var_19, dry=var_3)
    var_21 = module_0.gen_api(var_12, link=var_19, dry=var_3)
    var_22 = 'NonExistent'
    var_23 = 'nonexistent_module'
    var_24 = {var_22: var_23}
    var_25 = module_0.gen_api(var_24, dry=var_3)
    var_26 = len(var_25)
    var_27 = var_26 == var_19
    var_28 = True



# Parsed testcases at query #15
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_2, link=var_7, dry=var_3)
    var_9 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_10 = 2
    var_11 = module_0.gen_api(var_2, level=var_10, dry=var_3)
    var_12 = {}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = 'Module1'
    var_15 = 'Module2'
    var_16 = 'module_one'
    var_17 = 'module_two'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    var_21 = len(var_18)



# Parsed testcases at query #16
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_2, link=var_7, dry=var_3)
    var_9 = 2
    var_10 = module_0.gen_api(var_2, level=var_9, dry=var_3)
    var_11 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = module_0.gen_api(var_16, dry=var_3)
    var_18 = len(var_17)
    var_19 = len(var_16)
    var_20 = 'NonExistent'
    var_21 = 'nonexistent_module'
    var_22 = {var_20: var_21}
    var_23 = module_0.gen_api(var_22, dry=var_3)



# Parsed testcases at query #17
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\ndef test_func():\n    """Test function docstring."""\n    pass\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = True
    var_4 = 'test_package'
    var_5 = '.'
    var_6 = False
    var_7 = module_0.loader(var_4, var_5, var_6, var_3, var_6)
    var_8 = module_0.loader(var_4, var_5, var_3, var_3, var_6)
    var_9 = 2
    var_10 = module_0.loader(var_4, var_5, var_6, var_9, var_6)
    var_11 = module_0.loader(var_4, var_5, var_6, var_3, var_3)
    var_12 = 'non_existent_package'
    var_13 = module_0.loader(var_12, var_5, var_6, var_3, var_6)
    var_14 = 'no module'
    var_15 = 'empty_package'
    var_16 = module_0.loader(var_15, var_5, var_6, var_3, var_6)
    assert var_16 == ''



# Parsed testcases at query #18
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_2, link=var_7, dry=var_3)
    var_9 = 2
    var_10 = module_0.gen_api(var_2, level=var_9, dry=var_3)
    var_11 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_12 = {}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = 'Module1'
    var_15 = 'Module2'
    var_16 = 'module1'
    var_17 = 'module2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    var_21 = len(var_18)



# Parsed testcases at query #19
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = var_0 / var_5
    var_7 = '__init__.py'
    var_8 = var_6 / var_7
    var_9 = '"""Test module docstring."""\n'
    var_10 = 'module.py'
    var_11 = var_6 / var_10
    var_12 = '\n"""Module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_13 = 'docs'
    var_14 = False
    var_15 = 2
    var_16 = True
    var_17 = len(var_4)
    assert var_17 == 1
    var_18 = var_4[var_14]
    var_19 = 'test-module-api.md'
    var_20 = 'Empty'
    var_21 = 'non_existent_module'
    var_22 = {var_20: var_21}
    var_23 = module_0.gen_api(var_22, dry=var_16)
    var_24 = len(var_23)
    var_25 = var_24 == var_14
    var_26 = 'Module1'
    var_27 = 'Module2'
    var_28 = {var_26: var_5, var_27: var_5}
    var_29 = len(var_23)
    assert var_29 == 2
    var_30 = None
    var_31 = module_0.gen_api(var_2, var_30, dry=var_16)
    var_32 = 'new_docs'



# Parsed testcases at query #20
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = '__init__.py'
    var_7 = '"""Test module docstring."""\n'
    var_8 = 'TestModule'
    var_9 = {var_8: var_7}
    var_10 = False
    var_11 = 'test-module-api.md'
    var_12 = 'Module1'
    var_13 = 'Module2'
    var_14 = 'module1'
    var_15 = 'module2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = len(var_4)
    assert var_17 == 2
    var_18 = 'Test'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = {var_18: var_19}
    var_22 = 2
    var_23 = {var_18: var_19}
    var_24 = True
    var_25 = 'NonExistent'
    var_26 = 'nonexistent_module'
    var_27 = {var_25: var_26}
    var_28 = 'new_docs'
    var_29 = {var_18: var_19}
    var_30 = {}



# Parsed testcases at query #21
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestPackage'
    var_2 = 'test_package'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'custom_prefix'
    var_7 = True
    var_8 = 'Package1'
    var_9 = 'Package2'
    var_10 = 'package_one'
    var_11 = 'package_two'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_4)
    var_14 = len(var_13)
    var_15 = len(var_12)
    var_16 = False
    var_17 = module_0.gen_api(var_3, link=var_16, dry=var_4)
    var_18 = 2
    var_19 = module_0.gen_api(var_3, level=var_18, dry=var_4)
    var_20 = module_0.gen_api(var_3, toc=var_4, dry=var_4)
    var_21 = {}
    var_22 = module_0.gen_api(var_21, dry=var_4)
    var_23 = 'Fake'
    var_24 = 'non_existent_package_xyz'
    var_25 = {var_23: var_24}
    var_26 = module_0.gen_api(var_25, dry=var_4)
    var_27 = 'new_dir'
    var_28 = 'subdir'
    var_29 = True



# Parsed testcases at query #22
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module.py'
    var_2 = '\n"""Test module docstring."""\n\ndef test_function():\n    """Test function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    \n    def method(self):\n        """Test method docstring."""\n        pass\n'
    var_3 = True
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_4, var_0, var_5, var_5, var_6)
    var_8 = module_0.loader(var_4, var_0, var_6, var_5, var_6)
    var_9 = 2
    var_10 = module_0.loader(var_4, var_0, var_5, var_9, var_6)
    var_11 = module_0.loader(var_4, var_0, var_5, var_5, var_5)
    var_12 = 'nonexistent'
    var_13 = module_0.loader(var_12, var_0, var_5, var_5, var_6)
    var_14 = True
    var_15 = 'empty_test_dir'
    var_16 = 'test'
    var_17 = True
    var_18 = False
    var_19 = module_0.loader(var_16, var_15, var_17, var_17, var_18)
    assert var_19 == ''
    var_20 = True



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'test_module'
    var_6 = var_0 / var_5
    var_7 = '__init__.py'
    var_8 = var_6 / var_7
    var_9 = '"""Test module docstring."""'
    var_10 = 'simple.py'
    var_11 = var_6 / var_10
    var_12 = '\n"""Simple module."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_13 = 'docs'
    var_14 = False
    var_15 = 'test-module-api.md'
    var_16 = 'Module1'
    var_17 = 'Module2'
    var_18 = {var_16: var_5, var_17: var_5}
    var_19 = 'NotFound'
    var_20 = 'nonexistent_module'
    var_21 = {var_19: var_20}
    var_22 = True
    var_23 = module_0.gen_api(var_21, dry=var_22)
    var_24 = 2
    var_25 = None
    var_26 = module_0.gen_api(var_2, var_25, dry=var_22)



# Parsed testcases at query #25
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'docs'
    var_6 = False
    var_7 = 'test-module-api.md'
    var_8 = 'Module1'
    var_9 = 'Module2'
    var_10 = 'module1'
    var_11 = 'module2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_3)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = True
    var_16 = 'Test'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = False
    var_20 = 2
    var_21 = module_0.gen_api(var_18, link=var_19, level=var_20, toc=var_3, dry=var_3)
    var_22 = {}
    var_23 = module_0.gen_api(var_22, dry=var_3)
    var_24 = 'NonExistent'
    var_25 = 'nonexistent_module_xyz'
    var_26 = {var_24: var_25}
    var_27 = module_0.gen_api(var_26, dry=var_3)
    var_28 = 'new_docs'
    var_29 = True



# Parsed testcases at query #26
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'test_docs'
    var_7 = True
    var_8 = False
    var_9 = module_0.gen_api(var_3, link=var_8, dry=var_4)
    var_10 = 2
    var_11 = module_0.gen_api(var_3, level=var_10, dry=var_4)
    var_12 = 0
    var_13 = var_11[var_12]
    var_14 = '## '
    var_15 = module_0.gen_api(var_3, toc=var_4, dry=var_4)
    var_16 = {}
    var_17 = module_0.gen_api(var_16, dry=var_4)
    var_18 = 'Module1'
    var_19 = 'Module2'
    var_20 = 'module_one'
    var_21 = 'module_two'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = module_0.gen_api(var_22, dry=var_4)
    var_24 = len(var_23)
    var_25 = len(var_22)
    var_26 = 'Ghost'
    var_27 = 'non_existent_module_xyz'
    var_28 = {var_26: var_27}
    var_29 = module_0.gen_api(var_28, dry=var_4)
    var_30 = True
    var_31 = 3
    var_32 = module_0.gen_api(var_3, link=var_8, level=var_31, toc=var_4, dry=var_4)



# Parsed testcases at query #27
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = {}
    var_6 = module_0.gen_api(var_5)
    var_7 = 'NonExistent'
    var_8 = 'nonexistent_module'
    var_9 = {var_7: var_8}
    var_10 = module_0.gen_api(var_9, dry=var_3)
    var_11 = '/tmp'
    var_12 = module_0.gen_api(var_9, var_11, dry=var_3)
    var_13 = {var_0: var_1}
    var_14 = False
    var_15 = 2
    var_16 = module_0.gen_api(var_13, link=var_14, level=var_15, toc=var_3, dry=var_3)



# Parsed testcases at query #28
#--------------------------


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the gen_api function with various scenarios.'
    var_1 = 'TestPackage'
    var_2 = 'test_package'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'test_package'
    var_7 = '__init__.py'
    var_8 = '"""Test package."""\n'
    var_9 = True
    var_10 = len(var_5)
    assert var_10 == 1
    var_11 = 'custom_docs'
    var_12 = True
    var_13 = 'Package1'
    var_14 = 'Package2'
    var_15 = 'package_one'
    var_16 = 'package_two'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.gen_api(var_17, dry=var_9)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'Empty'
    var_21 = ''
    var_22 = {var_20: var_21}
    var_23 = module_0.gen_api(var_22, dry=var_9)
    var_24 = False
    var_25 = module_0.gen_api(var_22, link=var_24, dry=var_9)
    var_26 = 2
    var_27 = module_0.gen_api(var_22, level=var_26, dry=var_9)
    var_28 = module_0.gen_api(var_22, toc=var_9, dry=var_9)
    var_29 = 'NonExistent'
    var_30 = 'nonexistent_package_12345'
    var_31 = {var_29: var_30}
    var_32 = module_0.gen_api(var_31, dry=var_9)
    var_33 = True



# Parsed testcases at query #29
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
    var_6 = 'docs'
    var_7 = False
    var_8 = len(var_4)
    assert var_8 == 1
    var_9 = 'test-module-api.md'
    var_10 = 'Module1'
    var_11 = 'Module2'
    var_12 = 'module1'
    var_13 = 'module2'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = module_0.gen_api(var_14, dry=var_3)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = True
    var_18 = False
    var_19 = module_0.gen_api(var_14, link=var_18, dry=var_3)
    var_20 = 2
    var_21 = module_0.gen_api(var_14, level=var_20, dry=var_3)
    var_22 = module_0.gen_api(var_14, toc=var_3, dry=var_3)
    var_23 = {}
    var_24 = module_0.gen_api(var_23, dry=var_3)
    var_25 = 'NonExistent'
    var_26 = 'nonexistent_module'
    var_27 = {var_25: var_26}
    var_28 = module_0.gen_api(var_27, dry=var_3)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = var_28[var_18]



