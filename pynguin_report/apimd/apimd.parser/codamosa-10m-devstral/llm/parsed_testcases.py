####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = 'attr1'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = None
    var_9 = 1
    var_10 = 'attr2'
    var_11 = 42
    var_12 = 'attr3'



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid name'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a normal string.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a normal string.'
    var_2 = ">>> print('hello')"
    var_3 = module_0.doctest(var_2)
    assert var_3 == "```python\n>>> print('hello')\n```"
    var_4 = 'Example function:\n>>> def add(a, b):\n...     return a + b\n>>> add(1, 2)\n3'



# Parsed testcases at query #4
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Constant()
    var_6 = module_0.Assign()
    var_7 = [var_6]
    var_8 = module_1.walk_body(var_7)
    var_9 = list(var_8)
    var_10 = True
    var_11 = module_0.Constant()
    var_12 = 'y'
    var_13 = module_0.Load()
    var_14 = module_0.Name()
    var_15 = [var_14]
    var_16 = 2
    var_17 = module_0.Constant()
    var_18 = module_0.Assign()
    var_19 = [var_18]
    var_20 = 'z'
    var_21 = module_0.Load()
    var_22 = module_0.Name()
    var_23 = [var_22]
    var_24 = 3
    var_25 = module_0.Constant()
    var_26 = module_0.Assign()
    var_27 = [var_26]
    var_28 = module_0.If()
    var_29 = [var_28]
    var_30 = module_1.walk_body(var_29)
    var_31 = list(var_30)
    var_32 = module_0.Load()
    var_33 = module_0.Name()
    var_34 = [var_33]
    var_35 = module_0.Constant()
    var_36 = module_0.Assign()
    var_37 = module_0.Load()
    var_38 = module_0.Name()
    var_39 = [var_38]
    var_40 = module_0.Constant()
    var_41 = module_0.Assign()
    var_42 = [var_36, var_41]
    var_43 = 'a'
    var_44 = module_0.Load()
    var_45 = module_0.Name()
    var_46 = [var_45]
    var_47 = 4
    var_48 = module_0.Constant()
    var_49 = module_0.Assign()
    var_50 = [var_49]
    var_51 = []
    var_52 = 'b'
    var_53 = module_0.Load()
    var_54 = module_0.Name()
    var_55 = [var_54]
    var_56 = 5
    var_57 = module_0.Constant()
    var_58 = module_0.Assign()
    var_59 = [var_58]
    var_60 = 'c'
    var_61 = module_0.Load()
    var_62 = module_0.Name()
    var_63 = [var_62]
    var_64 = 6
    var_65 = module_0.Constant()
    var_66 = module_0.Assign()
    var_67 = [var_66]
    var_68 = module_0.Try()
    var_69 = [var_68]
    var_70 = module_1.walk_body(var_69)
    var_71 = list(var_70)
    var_72 = module_0.Load()
    var_73 = module_0.Name()
    var_74 = [var_73]
    var_75 = module_0.Constant()
    var_76 = module_0.Assign()
    var_77 = module_0.Load()
    var_78 = module_0.Name()
    var_79 = [var_78]
    var_80 = module_0.Constant()
    var_81 = module_0.Assign()
    var_82 = module_0.Load()
    var_83 = module_0.Name()
    var_84 = [var_83]
    var_85 = module_0.Constant()
    var_86 = module_0.Assign()
    var_87 = [var_76, var_81, var_86]
    var_88 = True
    var_89 = module_0.Constant()
    var_90 = 'd'
    var_91 = module_0.Load()
    var_92 = module_0.Name()
    var_93 = [var_92]
    var_94 = 7
    var_95 = module_0.Constant()
    var_96 = module_0.Assign()
    var_97 = False
    var_98 = module_0.Constant()
    var_99 = 'e'
    var_100 = module_0.Load()
    var_101 = module_0.Name()
    var_102 = [var_101]
    var_103 = 8
    var_104 = module_0.Constant()
    var_105 = module_0.Assign()
    var_106 = [var_105]
    var_107 = []
    var_108 = module_0.If()
    var_109 = [var_96, var_108]
    var_110 = []
    var_111 = module_0.If()
    var_112 = [var_111]
    var_113 = module_1.walk_body(var_112)
    var_114 = list(var_113)
    var_115 = module_0.Load()
    var_116 = module_0.Name()
    var_117 = [var_116]
    var_118 = module_0.Constant()
    var_119 = module_0.Assign()
    var_120 = module_0.Load()
    var_121 = module_0.Name()
    var_122 = [var_121]
    var_123 = module_0.Constant()
    var_124 = module_0.Assign()
    var_125 = [var_119, var_124]
    var_126 = []
    var_127 = module_1.walk_body(var_126)
    var_128 = list(var_127)



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR1'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 'VAR2'
    var_8 = 'hello'
    var_9 = 'str'
    var_10 = 'VAR3'
    var_11 = 1
    var_12 = 2
    var_13 = '__all__'
    var_14 = 'CONST'
    var_15 = module_1.Load()
    var_16 = 'float'
    var_17 = module_1.Load()
    var_18 = 3.14
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 10



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'hello_world'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == 'hello\\_world'
    var_2 = 'single_'
    var_3 = module_0.esc_underscore(var_2)
    assert var_3 == 'single_'
    var_4 = 'no_underscore'
    var_5 = module_0.esc_underscore(var_4)
    assert var_5 == 'no_underscore'
    var_6 = '__double__'
    var_7 = module_0.esc_underscore(var_6)
    assert var_7 == '\\_\\_double\\_\\_'
    var_8 = ''
    var_9 = module_0.esc_underscore(var_8)
    assert var_9 == ''



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = 'y'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = 'return'
    var_8 = 'bool'
    var_9 = module_1.Load()
    var_10 = 'module'
    var_11 = False
    var_12 = 'self'
    var_13 = 'Class'
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = True
    var_18 = None
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = '*'
    var_22 = module_1.Load()
    var_23 = '**'
    var_24 = module_1.Load()
    var_25 = 'List'
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = 'Dict'
    var_30 = module_1.Load()
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = module_0.Parser()
    var_7 = 'collections'
    var_8 = 'defaultdict'
    var_9 = 0
    var_10 = module_0.Parser()
    var_11 = 'submodule'
    var_12 = 'func'
    var_13 = 'f'
    var_14 = 1
    var_15 = module_0.Parser()



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.simple_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_1, var_2, var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = '| | |\n| --- | --- |\n| return | Any |'
    var_14 = module_0.Parser()
    var_15 = 'test_module'
    var_16 = 'test_module.func_with_args'
    var_17 = 'arg1'
    var_18 = None
    var_19 = 'arg2'
    var_20 = 'arg3'
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'int'
    var_25 = module_1.Load()
    var_26 = False
    var_27 = False
    var_28 = var_14.func_api(var_15, var_16, var_8, var_9, has_self=var_26, cls_method=var_27)
    var_29 = '| arg1 | arg2 | / | arg3 | return |\n| --- | --- | --- | --- | --- |\n| Any | Any |  | Any | int |'
    var_30 = module_0.Parser()
    var_31 = 'test_module'
    var_32 = 'test_module.func_with_defaults'
    var_33 = []
    var_34 = []
    var_35 = []
    var_36 = 1
    var_37 = 2
    var_38 = None
    var_39 = False
    var_40 = False
    var_41 = var_30.func_api(var_31, var_32, var_8, var_38, has_self=var_39, cls_method=var_40)
    var_42 = '| arg1 | arg2 | return |\n| --- | --- | --- |\n| Any | Any | Any |\n| 1 | 2 |  |'
    var_43 = module_0.Parser()
    var_44 = 'test_module'
    var_45 = 'test_module.MyClass.method'
    var_46 = []
    var_47 = 'self'
    var_48 = []
    var_49 = []
    var_50 = []
    var_51 = None
    var_52 = True
    var_53 = False
    var_54 = var_43.func_api(var_44, var_45, var_8, var_51, has_self=var_52, cls_method=var_53)
    var_55 = '| self | arg1 | return |\n| --- | --- | --- |\n| Self | Any | Any |'
    var_56 = module_0.Parser()
    var_57 = 'test_module'
    var_58 = 'test_module.MyClass.class_method'
    var_59 = []
    var_60 = 'cls'
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = None
    var_65 = True
    var_66 = True
    var_67 = var_56.func_api(var_57, var_58, var_8, var_64, has_self=var_65, cls_method=var_66)
    var_68 = '| cls | arg1 | return |\n| --- | --- | --- |\n| type[Self] | Any | Any |'
    var_69 = module_0.Parser()
    var_70 = 'test_module'
    var_71 = 'test_module.func_with_varargs'
    var_72 = []
    var_73 = []
    var_74 = []
    var_75 = []
    var_76 = 'args'
    var_77 = 'kwargs'
    var_78 = None
    var_79 = False
    var_80 = False
    var_81 = var_69.func_api(var_70, var_71, var_8, var_78, has_self=var_79, cls_method=var_80)
    var_82 = '| arg1 | *args | **kwargs | return |\n| --- | --- | --- | --- |\n| Any |  |  | Any |'
    var_83 = module_0.Parser()
    var_84 = 'test_module'
    var_85 = 'test_module.func_with_kwonly'
    var_86 = []
    var_87 = 'kwarg1'
    var_88 = 'kwarg2'
    var_89 = []
    var_90 = None
    var_91 = False
    var_92 = False
    var_93 = var_83.func_api(var_84, var_85, var_8, var_90, has_self=var_91, cls_method=var_92)
    var_94 = '| arg1 | * | kwarg1 | kwarg2 | return |\n| --- | --- | --- | --- | --- |\n| Any |  | Any | Any | Any |\n|  |  | 1 | 2 |  |'



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'invalid expression'
    var_5 = 'test_var'
    var_6 = 'test.test_var'
    var_7 = 'alias_var'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'public_name'
    var_3 = 'parent.child'
    var_4 = {var_2, var_3}
    var_5 = 'module.public_name'
    var_6 = 'module.parent.child'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    assert var_8 is True
    var_9 = var_0.is_public(var_6)
    assert var_9 is True
    var_10 = module_0.Parser()
    var_11 = set()
    var_12 = 'module._private_name'
    var_13 = var_10.is_public(var_12)
    assert var_13 is False
    var_14 = module_0.Parser()
    var_15 = set()
    var_16 = 'module._internal'
    var_17 = var_14.is_public(var_16)
    assert var_17 is False
    var_18 = module_0.Parser()
    var_19 = '_in_all'
    var_20 = {var_19}
    var_21 = 'module._in_all'
    var_22 = var_18.is_public(var_21)
    assert var_22 is True
    var_23 = module_0.Parser()
    var_24 = 'parent'
    var_25 = {var_24}
    var_26 = var_23.is_public(var_6)
    assert var_26 is True
    var_27 = module_0.Parser()
    var_28 = set()
    var_29 = 'module._private.parent.child'
    var_30 = var_27.is_public(var_29)
    assert var_30 is False



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = 'attr1'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = None
    var_9 = 1
    var_10 = 'attr2'
    var_11 = 42
    var_12 = 'float'
    var_13 = 'attr3'



# Parsed testcases at query #13
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = '\nclass TestClass:\n    \'\'\'This is a test class.\'\'\'\n    x: int\n    y: str = "hello"\n    def __init__(self):\n        pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test_module.TestClass'
    var_6 = []



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = 'x'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = module_1.Assign()
    var_10 = [var_9]
    var_11 = module_0.walk_body(var_10)
    var_12 = list(var_11)
    var_13 = True
    var_14 = module_1.Constant()
    var_15 = 'y'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = [var_17]
    var_19 = 2
    var_20 = module_1.Constant()
    var_21 = module_1.Assign()
    var_22 = [var_21]
    var_23 = []
    var_24 = module_1.If()
    var_25 = [var_24]
    var_26 = module_0.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'z'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = [var_31]
    var_33 = 3
    var_34 = module_1.Constant()
    var_35 = module_1.Assign()
    var_36 = [var_35]
    var_37 = 'type'
    var_38 = 'body'
    var_39 = 'Exception'
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = 'handler1'
    var_43 = module_1.Constant()
    var_44 = module_1.Expr()
    var_45 = [var_44]
    var_46 = {var_37: var_41, var_38: var_45}
    var_47 = 'ValueError'
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = 'handler2'
    var_51 = module_1.Constant()
    var_52 = module_1.Expr()
    var_53 = [var_52]
    var_54 = {var_37: var_49, var_38: var_53}
    var_55 = [var_46, var_54]
    var_56 = 'else'
    var_57 = module_1.Constant()
    var_58 = module_1.Expr()
    var_59 = [var_58]
    var_60 = 'finally'
    var_61 = module_1.Constant()
    var_62 = module_1.Expr()
    var_63 = [var_62]
    var_64 = module_1.Try()
    var_65 = [var_64]
    var_66 = module_0.walk_body(var_65)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 4
    var_69 = 0
    var_70 = var_67[var_69]
    var_71 = var_67[var_13]
    var_72 = var_67[var_19]
    var_73 = var_67[var_33]
    var_74 = 'a'
    var_75 = module_1.Load()
    var_76 = module_1.Name()
    var_77 = [var_76]
    var_78 = module_1.Constant()
    var_79 = module_1.Assign()
    var_80 = True
    var_81 = module_1.Constant()
    var_82 = 'b'
    var_83 = module_1.Load()
    var_84 = module_1.Name()
    var_85 = [var_84]
    var_86 = module_1.Constant()
    var_87 = module_1.Assign()
    var_88 = 'c'
    var_89 = module_1.Load()
    var_90 = module_1.Name()
    var_91 = [var_90]
    var_92 = module_1.Constant()
    var_93 = module_1.Assign()
    var_94 = [var_93]
    var_95 = []
    var_96 = []
    var_97 = []
    var_98 = module_1.Try()
    var_99 = [var_87, var_98]
    var_100 = []
    var_101 = module_1.If()
    var_102 = [var_79, var_101]
    var_103 = module_0.walk_body(var_102)
    var_104 = list(var_103)
    var_105 = len(var_104)
    assert var_105 == 3



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.A'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 123
    var_6 = 'not_a_name'
    var_7 = 'A'
    var_8 = {}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = 'y'
    var_6 = 'str'
    var_7 = module_1.Load()
    var_8 = 'return'
    var_9 = 'bool'
    var_10 = module_1.Load()
    var_11 = False
    var_12 = 'self'
    var_13 = 'TestClass'
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = True
    var_18 = 'cls'
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = '*'
    var_24 = None
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Load()
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = 1
    var_6 = 'test'
    var_7 = 'y'
    var_8 = module_1.Load()
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = None
    var_12 = 'z'
    var_13 = 3.14
    var_14 = 'float'
    var_15 = 'w'
    var_16 = 2
    var_17 = '__all__'
    var_18 = 'public_func'
    var_19 = 'PublicClass'
    var_20 = 'arr'
    var_21 = 0
    var_22 = 42
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 10



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'numpy'
    var_5 = 'np'
    var_6 = 'sys'
    var_7 = 'path'
    var_8 = 0
    var_9 = 'collections'
    var_10 = 'defaultdict'
    var_11 = 'dd'
    var_12 = 1
    var_13 = 'operating_system'



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = 'a'
    var_4 = None
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = 1
    var_9 = 2
    var_10 = 'e'
    var_11 = 'f'
    var_12 = 3
    var_13 = 4
    var_14 = 'result'
    var_15 = False



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'y'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = 'z'
    var_14 = module_1.Constant()
    var_15 = 2
    var_16 = module_1.Constant()
    var_17 = [var_14, var_16]
    var_18 = module_1.List()
    var_19 = '__all__'
    var_20 = 'public_func'
    var_21 = module_1.Constant()
    var_22 = 'public_var'
    var_23 = module_1.Constant()
    var_24 = [var_21, var_23]
    var_25 = module_1.List()
    var_26 = 'non_const'
    var_27 = 'some_func'
    var_28 = module_1.Load()
    var_29 = module_1.Name()
    var_30 = []
    var_31 = module_1.Call(*var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = 10
    var_35 = module_1.Constant()



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = 'test_module.VAR'
    var_9 = module_0.Parser()
    var_10 = 'CONST'
    var_11 = module_1.Load()
    var_12 = 3.14
    var_13 = 'float'
    var_14 = 'test_module.CONST'
    var_15 = module_0.Parser()
    var_16 = 'LIST'
    var_17 = module_1.Load()
    var_18 = 2
    var_19 = module_1.Load()
    var_20 = 'test_module.LIST'
    var_21 = module_0.Parser()
    var_22 = '__all__'
    var_23 = module_1.Load()
    var_24 = 'public_func'
    var_25 = 'PublicClass'
    var_26 = module_1.Load()
    var_27 = module_0.Parser()
    var_28 = 'non_const'
    var_29 = module_1.Load()
    var_30 = 'some_var'
    var_31 = module_1.Load()
    var_32 = 'test_module.non_const'
    var_33 = module_0.Parser()
    var_34 = 'a'
    var_35 = module_1.Load()
    var_36 = 'b'
    var_37 = module_1.Load()
    var_38 = 10



# Parsed testcases at query #22
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.typing.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union'
    var_6 = module_1.Load()
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = module_1.Load()
    var_13 = 'Optional'
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'test.typing.List'
    var_18 = 'typing.List'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = 'List'
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = module_1.Load()
    var_25 = 'SomeClass'
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = module_1.Load()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = module_0.is_public_family(var_0)
    var_2 = 'module.submodule.Class'
    var_3 = module_0.is_public_family(var_2)
    var_4 = 'module.submodule.Class.method'
    var_5 = module_0.is_public_family(var_4)
    var_6 = 'module.submodule.Class.__init__'
    var_7 = module_0.is_public_family(var_6)
    var_8 = 'module.submodule.__class__'
    var_9 = module_0.is_public_family(var_8)
    var_10 = '_private_module'
    var_11 = module_0.is_public_family(var_10)
    var_12 = 'module._private_submodule'
    var_13 = module_0.is_public_family(var_12)
    var_14 = 'module.submodule._private_class'
    var_15 = module_0.is_public_family(var_14)
    var_16 = 'module.submodule.Class._private_method'
    var_17 = module_0.is_public_family(var_16)
    var_18 = 'module.submodule.Class.__private_magic__'
    var_19 = module_0.is_public_family(var_18)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR1'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 'VAR2'
    var_8 = 'hello'
    var_9 = 'str'
    var_10 = 'VAR3'
    var_11 = 1
    var_12 = 2
    var_13 = '__all__'
    var_14 = 'public_func'
    var_15 = 'public_var'
    var_16 = 'non_const'
    var_17 = 'some_func'
    var_18 = module_1.Load()
    var_19 = []
    var_20 = 'a'
    var_21 = 'b'
    var_22 = 10



# Parsed testcases at query #3
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = 'def func(): pass'
    var_3 = module_0.parse(var_1, var_2)
    var_4 = False
    var_5 = 2
    var_6 = '"""\n    This is a test module.\n    """'
    var_7 = module_0.parse(var_1, var_6)
    var_8 = '\nclass TestClass:\n    """A test class."""\n    def method(self):\n        """A test method."""\n        pass\n'
    var_9 = module_0.parse(var_1, var_8)
    var_10 = '\nfrom typing import List\nimport os\n\ndef func() -> List[int]:\n    pass\n'
    var_11 = module_0.parse(var_1, var_10)
    var_12 = '\nCONSTANT = 42\n'
    var_13 = module_0.parse(var_1, var_12)
    var_14 = '\ndef _private_func():\n    pass\n\nclass TestClass:\n    def _private_method(self):\n        pass\n'
    var_15 = module_0.parse(var_1, var_14)
    var_16 = "\n__all__ = ['public_func']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n"
    var_17 = module_0.parse(var_1, var_16)



# Parsed testcases at query #4
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = module_1.arguments(*var_9)
    var_14 = []
    var_15 = []
    var_16 = 'str'
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = module_1.FunctionDef(*var_13)
    var_20 = var_0.api(var_1, var_19)
    var_21 = '## test_func()\n\n*Full name:* `test_module.test_func`\n<a id="test_module-test_func"></a>\n\n|  |   |\n|---|---|\n| **x** | `int` |\n| **return** | `str` |'
    var_22 = 'test_module.test_func'
    var_23 = var_0.doc[var_22]
    var_24 = module_0.Parser()
    var_25 = 'async_test'
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = []
    var_30 = []
    var_31 = module_1.arguments(*var_27)
    var_32 = []
    var_33 = []
    var_34 = 'None'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = module_1.AsyncFunctionDef(*var_31)
    var_38 = 'TestClass'
    var_39 = var_24.api(var_1, var_37, prefix=var_38)
    var_40 = '#### async TestClass.async_test()\n\n*Full name:* `test_module.TestClass.async_test`\n<a id="test_module-TestClass-async_test"></a>\n\n|  |   |\n|---|---|\n| **return** | `None` |'
    var_41 = 'test_module.TestClass.async_test'
    var_42 = var_24.doc[var_41]
    var_43 = module_0.Parser()
    var_44 = 'BaseClass'
    var_45 = module_1.Load()
    var_46 = module_1.Name()
    var_47 = [var_46]
    var_48 = 'attr1'
    var_49 = module_1.Load()
    var_50 = module_1.Name()
    var_51 = 1
    var_52 = module_1.Constant()
    var_53 = 'method1'
    var_54 = []
    var_55 = []
    var_56 = []
    var_57 = []
    var_58 = []
    var_59 = module_1.arguments(*var_55)
    var_60 = []
    var_61 = []
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = module_1.FunctionDef(*var_59)
    var_65 = []
    var_66 = var_43.api(var_1, var_37)
    var_67 = '## class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id="test_module-TestClass"></a>\n\n**Bases:** `BaseClass`\n\n| Members | Type |\n|---------|------|\n| `attr1` | `int` |'
    var_68 = 'test_module.TestClass'
    var_69 = var_43.doc[var_68]
    var_70 = module_0.Parser()
    var_71 = 'decorated_func'
    var_72 = []
    var_73 = []
    var_74 = []
    var_75 = []
    var_76 = []
    var_77 = module_1.arguments(*var_73)
    var_78 = []
    var_79 = 'staticmethod'
    var_80 = module_1.Load()
    var_81 = module_1.Name()
    var_82 = 'custom_decorator'
    var_83 = module_1.Load()
    var_84 = module_1.Name()
    var_85 = [var_81, var_84]
    var_86 = module_1.Load()
    var_87 = module_1.Name()
    var_88 = module_1.FunctionDef(*var_77)
    var_89 = var_70.api(var_1, var_88)
    var_90 = '## decorated_func()\n\n*Full name:* `test_module.decorated_func`\n<a id="test_module-decorated_func"></a>\n\n**Decorators:**\n|  |   |\n|---|---|\n| `@staticmethod` |  |\n| `@custom_decorator` |  |'
    var_91 = 'test_module.decorated_func'
    var_92 = var_70.doc[var_91]



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'test.typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'test.typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = {}
    var_29 = module_0.Resolver(var_0, var_28)
    var_30 = 'SomeClass'
    var_31 = module_1.Load()
    var_32 = 'attr'
    var_33 = module_1.Load()
    var_34 = module_1.Load()



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'test_module.test_function'
    var_4 = '# Module `test_module`'
    var_5 = '## class TestClass'
    var_6 = '## test_function()'
    var_7 = 'Module docstring'
    var_8 = type('TestClass', (), {'__doc__': 'Class docstring'})
    var_9 = lambda : None



# Parsed testcases at query #7
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'simple_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_0.arguments(*var_5)
    var_10 = None
    var_11 = []
    var_12 = module_0.FunctionDef(*var_9)
    var_13 = 'test_module.simple_func'
    var_14 = var_12.args
    var_15 = var_12.returns
    var_16 = 'func_with_args'
    var_17 = 'x'
    var_18 = 'int'
    var_19 = module_0.Load()
    var_20 = module_0.Name()
    var_21 = module_0.arg()
    var_22 = [var_21]
    var_23 = 'y'
    var_24 = 'str'
    var_25 = module_0.Load()
    var_26 = module_0.Name()
    var_27 = module_0.arg()
    var_28 = [var_27]
    var_29 = []
    var_30 = []
    var_31 = []
    var_32 = module_0.arguments(*var_28)
    var_33 = 'bool'
    var_34 = module_0.Load()
    var_35 = module_0.Name()
    var_36 = []
    var_37 = module_0.FunctionDef(*var_32)
    var_38 = 'test_module.func_with_args'
    var_39 = var_37.args
    var_40 = var_37.returns
    var_41 = 'func_with_defaults'
    var_42 = []
    var_43 = 'a'
    var_44 = module_0.Load()
    var_45 = module_0.Name()
    var_46 = module_0.arg()
    var_47 = [var_46]
    var_48 = []
    var_49 = []
    var_50 = module_0.Constant()
    var_51 = [var_50]
    var_52 = module_0.arguments(*var_47)
    var_53 = []
    var_54 = module_0.FunctionDef(*var_52)
    var_55 = 'test_module.func_with_defaults'
    var_56 = var_54.args
    var_57 = var_54.returns
    var_58 = 'func_with_varargs'
    var_59 = []
    var_60 = []
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = 'args'
    var_65 = module_0.arg()
    var_66 = 'kwargs'
    var_67 = module_0.arg()
    var_68 = module_0.arguments(*var_60)
    var_69 = []
    var_70 = module_0.FunctionDef(*var_68)
    var_71 = 'test_module.func_with_varargs'
    var_72 = var_70.args
    var_73 = var_70.returns
    var_74 = 'method'
    var_75 = []
    var_76 = 'self'
    var_77 = module_0.arg()
    var_78 = [var_77]
    var_79 = []
    var_80 = []
    var_81 = []
    var_82 = module_0.arguments(*var_78)
    var_83 = []
    var_84 = module_0.FunctionDef(*var_82)
    var_85 = 'test_module.method'
    var_86 = var_84.args
    var_87 = var_84.returns
    var_88 = True
    var_89 = 'cls_method'
    var_90 = []
    var_91 = 'cls'
    var_92 = module_0.arg()
    var_93 = [var_92]
    var_94 = []
    var_95 = []
    var_96 = []
    var_97 = module_0.arguments(*var_93)
    var_98 = 'classmethod'
    var_99 = module_0.Load()
    var_100 = module_0.Name()
    var_101 = [var_100]
    var_102 = module_0.FunctionDef(*var_97)
    var_103 = 'test_module.cls_method'
    var_104 = var_102.args
    var_105 = var_102.returns
    var_106 = True



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = module_1.Expr()
    var_6 = 'x'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = 2
    var_11 = module_1.Constant()
    var_12 = module_1.Assign()
    var_13 = [var_5, var_12]
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = [var_5]
    var_19 = [var_12]
    var_20 = module_1.If()
    var_21 = [var_20]
    var_22 = module_0.walk_body(var_21)
    var_23 = list(var_22)
    var_24 = [var_5]
    var_25 = []
    var_26 = [var_12]
    var_27 = []
    var_28 = module_1.Try()
    var_29 = [var_28]
    var_30 = module_0.walk_body(var_29)
    var_31 = list(var_30)
    var_32 = True
    var_33 = module_1.Constant()
    var_34 = False
    var_35 = module_1.Constant()
    var_36 = [var_5]
    var_37 = [var_12]
    var_38 = module_1.If()
    var_39 = [var_38]
    var_40 = []
    var_41 = module_1.If()
    var_42 = [var_41]
    var_43 = module_0.walk_body(var_42)
    var_44 = list(var_43)
    var_45 = 3
    var_46 = module_1.Constant()
    var_47 = module_1.Expr()
    var_48 = [var_5]
    var_49 = 'type'
    var_50 = 'name'
    var_51 = 'body'
    var_52 = None
    var_53 = [var_47]
    var_54 = {var_49: var_52, var_50: var_52, var_51: var_53}
    var_55 = [var_54]
    var_56 = []
    var_57 = []
    var_58 = module_1.Try()
    var_59 = [var_58]
    var_60 = module_0.walk_body(var_59)
    var_61 = list(var_60)



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_0.Parser()
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = module_0.Parser()
    var_14 = module_1.Constant()
    var_15 = 2
    var_16 = module_1.Constant()
    var_17 = [var_14, var_16]
    var_18 = module_1.List()
    var_19 = module_0.Parser()
    var_20 = '__all__'
    var_21 = 'public_func'
    var_22 = module_1.Constant()
    var_23 = 'public_class'
    var_24 = module_1.Constant()
    var_25 = [var_22, var_24]
    var_26 = module_1.List()
    var_27 = module_0.Parser()
    var_28 = 'non_const'
    var_29 = 'some_var'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_0.Parser()
    var_33 = 'var1'
    var_34 = 'var2'
    var_35 = 10
    var_36 = module_1.Constant()



# Parsed testcases at query #10
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = "\ndef test_function():\n    '''This is a test function.'''\n    pass\n"
    var_4 = module_0.parse(var_2, var_3)
    var_5 = '## test_function()\n*Full name:* `test_module.test_function`\n\n'
    var_6 = 'test_module'
    var_7 = "\nclass TestClass:\n    async def test_method(self):\n        '''This is a test method.'''\n        pass\n"
    var_8 = module_0.parse(var_6, var_7)
    var_9 = '### async test_method()\n*Full name:* `test_module.TestClass.test_method`\n\n'
    var_10 = 'test_module'
    var_11 = "\n@decorator1\n@decorator2\nclass TestClass:\n    '''This is a test class.'''\n    pass\n"
    var_12 = module_0.parse(var_10, var_11)
    var_13 = '## class TestClass\n*Full name:* `test_module.TestClass`\n\n| Decorators |\n|------------|\n| `@decorator1` |\n| `@decorator2` |\n'
    var_14 = 'test_module'
    var_15 = "\ndef test_function() -> int:\n    '''This is a test function.'''\n    return 1\n"
    var_16 = module_0.parse(var_14, var_15)
    var_17 = '## test_function()\n*Full name:* `test_module.test_function`\n\n|  |  |\n|---|---|\n| return | `int` |\n'
    var_18 = 'test_module'
    var_19 = "\nclass TestClass(BaseClass):\n    '''This is a test class.'''\n    pass\n"
    var_20 = module_0.parse(var_18, var_19)
    var_21 = '## class TestClass\n*Full name:* `test_module.TestClass`\n\n| Bases |\n|-------|\n| `BaseClass` |\n'
    var_22 = 'test_module'
    var_23 = "\nclass OuterClass:\n    class InnerClass:\n        '''This is an inner class.'''\n        pass\n"
    var_24 = module_0.parse(var_22, var_23)
    var_25 = '### class InnerClass\n*Full name:* `test_module.OuterClass.InnerClass`\n\n'



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'public_name'
    var_3 = 'another_name'
    var_4 = {var_2, var_3}
    var_5 = 'module.public_name'
    var_6 = 'module._private'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    assert var_8 is True
    var_9 = var_0.is_public(var_6)
    assert var_9 is False
    var_10 = module_0.Parser()
    var_11 = 'listed_name'
    var_12 = {var_11}
    var_13 = 'module.listed_name'
    var_14 = var_10.is_public(var_13)
    assert var_14 is True
    var_15 = module_0.Parser()
    var_16 = set()
    var_17 = var_15.is_public(var_6)
    assert var_17 is False
    var_18 = module_0.Parser()
    var_19 = 'parent'
    var_20 = 'parent.child'
    var_21 = {var_20}
    var_22 = 'parent.child.sub'
    var_23 = var_18.is_public(var_22)
    assert var_23 is True
    var_24 = module_0.Parser()
    var_25 = set()
    var_26 = 'module._private_sub'
    var_27 = var_24.is_public(var_26)
    assert var_27 is False
    var_28 = module_0.Parser()
    var_29 = {var_1}
    var_30 = var_28.is_public(var_1)
    assert var_30 is True



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.a'
    var_2 = 'test.b'
    var_3 = 'int'
    var_4 = 'str'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'Self'
    var_8 = module_1.Load()
    var_9 = 'a'
    var_10 = module_1.Load()
    var_11 = 'c'
    var_12 = module_1.Load()



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid name!'
    var_7 = 'test.ValidName'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'ValidName'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = 'typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Load()
    var_14 = 'str'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'typing.Optional'
    var_18 = 'Optional'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'typing.Dict'
    var_24 = 'dict'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = 'Dict'
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'not_a_valid_name!'
    var_5 = 'valid_name'
    var_6 = 'some.module.name'



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = 'y'
    var_6 = 'str'
    var_7 = module_1.Load()
    var_8 = 'return'
    var_9 = 'bool'
    var_10 = module_1.Load()
    var_11 = False
    var_12 = 'self'
    var_13 = 'TestClass'
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = True
    var_18 = 'cls'
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = '*'
    var_24 = None
    var_25 = module_1.Load()
    var_26 = '**'
    var_27 = 'kwargs'
    var_28 = 'dict'
    var_29 = module_1.Load()
    var_30 = module_1.Load()
    var_31 = module_1.Load()
    var_32 = module_1.Load()



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'test_module'
    var_3 = module_0.Parser()
    var_4 = 'numpy'
    var_5 = 'np'
    var_6 = module_0.Parser()
    var_7 = 'sys'
    var_8 = 'path'
    var_9 = 0
    var_10 = module_0.Parser()
    var_11 = 'osp'
    var_12 = 1
    var_13 = 'test_module.submodule'



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = 'x'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = module_1.Assign()
    var_10 = 'print'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = []
    var_14 = []
    var_15 = module_1.Call(*var_13)
    var_16 = module_1.Expr()
    var_17 = [var_9, var_16]
    var_18 = module_0.walk_body(var_17)
    var_19 = list(var_18)
    var_20 = True
    var_21 = module_1.Constant()
    var_22 = 'y'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = [var_24]
    var_26 = 2
    var_27 = module_1.Constant()
    var_28 = module_1.Assign()
    var_29 = [var_28]
    var_30 = 'exit'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = []
    var_34 = []
    var_35 = module_1.Call(*var_33)
    var_36 = module_1.Expr()
    var_37 = [var_36]
    var_38 = module_1.If()
    var_39 = [var_38]
    var_40 = module_0.walk_body(var_39)
    var_41 = list(var_40)
    var_42 = module_1.Load()
    var_43 = module_1.Name()
    var_44 = [var_43]
    var_45 = module_1.Constant()
    var_46 = module_1.Assign()
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = []
    var_50 = []
    var_51 = module_1.Call(*var_49)
    var_52 = module_1.Expr()
    var_53 = [var_46, var_52]
    var_54 = 'z'
    var_55 = module_1.Load()
    var_56 = module_1.Name()
    var_57 = [var_56]
    var_58 = 3
    var_59 = module_1.Constant()
    var_60 = module_1.Assign()
    var_61 = [var_60]
    var_62 = []
    var_63 = 'log'
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = []
    var_67 = []
    var_68 = module_1.Call(*var_66)
    var_69 = module_1.Expr()
    var_70 = [var_69]
    var_71 = 'cleanup'
    var_72 = module_1.Load()
    var_73 = module_1.Name()
    var_74 = []
    var_75 = []
    var_76 = module_1.Call(*var_74)
    var_77 = module_1.Expr()
    var_78 = [var_77]
    var_79 = module_1.Try()
    var_80 = [var_79]
    var_81 = module_0.walk_body(var_80)
    var_82 = list(var_81)
    var_83 = module_1.Load()
    var_84 = module_1.Name()
    var_85 = [var_84]
    var_86 = module_1.Constant()
    var_87 = module_1.Assign()
    var_88 = module_1.Load()
    var_89 = module_1.Name()
    var_90 = []
    var_91 = []
    var_92 = module_1.Call(*var_90)
    var_93 = module_1.Expr()
    var_94 = module_1.Load()
    var_95 = module_1.Name()
    var_96 = []
    var_97 = []
    var_98 = module_1.Call(*var_96)
    var_99 = module_1.Expr()
    var_100 = [var_87, var_93, var_99]
    var_101 = True
    var_102 = module_1.Constant()
    var_103 = 'a'
    var_104 = module_1.Load()
    var_105 = module_1.Name()
    var_106 = [var_105]
    var_107 = 4
    var_108 = module_1.Constant()
    var_109 = module_1.Assign()
    var_110 = [var_109]
    var_111 = []
    var_112 = []
    var_113 = []
    var_114 = module_1.Try()
    var_115 = [var_114]
    var_116 = []
    var_117 = module_1.If()
    var_118 = [var_117]
    var_119 = module_0.walk_body(var_118)
    var_120 = list(var_119)
    var_121 = module_1.Load()
    var_122 = module_1.Name()
    var_123 = [var_122]
    var_124 = module_1.Constant()
    var_125 = module_1.Assign()
    var_126 = [var_125]
    var_127 = 'b'
    var_128 = module_1.Load()
    var_129 = module_1.Name()
    var_130 = [var_129]
    var_131 = 5
    var_132 = module_1.Constant()
    var_133 = module_1.Assign()
    var_134 = True
    var_135 = module_1.Constant()
    var_136 = 'func'
    var_137 = module_1.Load()
    var_138 = module_1.Name()
    var_139 = []
    var_140 = []
    var_141 = module_1.Call(*var_139)
    var_142 = module_1.Expr()
    var_143 = [var_142]
    var_144 = []
    var_145 = module_1.If()
    var_146 = []
    var_147 = []
    var_148 = 'c'
    var_149 = module_1.Load()
    var_150 = module_1.Name()
    var_151 = [var_150]
    var_152 = 6
    var_153 = module_1.Constant()
    var_154 = module_1.Assign()
    var_155 = [var_154]
    var_156 = []
    var_157 = module_1.Try()
    var_158 = [var_133, var_145, var_157]
    var_159 = module_0.walk_body(var_158)
    var_160 = list(var_159)
    var_161 = module_1.Load()
    var_162 = module_1.Name()
    var_163 = [var_162]
    var_164 = module_1.Constant()
    var_165 = module_1.Assign()
    var_166 = module_1.Load()
    var_167 = module_1.Name()
    var_168 = []
    var_169 = []
    var_170 = module_1.Call(*var_168)
    var_171 = module_1.Expr()
    var_172 = module_1.Load()
    var_173 = module_1.Name()
    var_174 = [var_173]
    var_175 = module_1.Constant()
    var_176 = module_1.Assign()
    var_177 = [var_165, var_171, var_176]



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'simple_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = []
    var_10 = []
    var_11 = module_1.FunctionDef(*var_8)
    var_12 = 'test_module.simple_func'
    var_13 = var_11.args
    var_14 = None
    var_15 = False
    var_16 = var_0.func_api(var_1, var_12, var_13, var_14, has_self=var_15, cls_method=var_15)
    var_17 = 'pos_args_func'
    var_18 = 'arg1'
    var_19 = 'int'
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = module_1.arg()
    var_23 = [var_22]
    var_24 = 'arg2'
    var_25 = 'str'
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.arg()
    var_29 = [var_28]
    var_30 = []
    var_31 = []
    var_32 = []
    var_33 = module_1.arguments(*var_29)
    var_34 = []
    var_35 = []
    var_36 = module_1.FunctionDef(*var_33)
    var_37 = 'test_module.pos_args_func'
    var_38 = var_36.args
    var_39 = var_0.func_api(var_1, var_37, var_38, var_14, has_self=var_15, cls_method=var_15)
    var_40 = 'kw_args_func'
    var_41 = []
    var_42 = []
    var_43 = 'kw_arg'
    var_44 = 'float'
    var_45 = module_1.Load()
    var_46 = module_1.Name()
    var_47 = module_1.arg()
    var_48 = [var_47]
    var_49 = []
    var_50 = []
    var_51 = module_1.arguments(*var_42)
    var_52 = []
    var_53 = []
    var_54 = module_1.FunctionDef(*var_51)
    var_55 = 'test_module.kw_args_func'
    var_56 = var_54.args
    var_57 = var_0.func_api(var_1, var_55, var_56, var_14, has_self=var_15, cls_method=var_15)
    var_58 = 'varargs_func'
    var_59 = []
    var_60 = []
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = 'args'
    var_65 = module_1.arg()
    var_66 = 'kwargs'
    var_67 = module_1.arg()
    var_68 = module_1.arguments(*var_60)
    var_69 = []
    var_70 = []
    var_71 = module_1.FunctionDef(*var_68)
    var_72 = 'test_module.varargs_func'
    var_73 = var_71.args
    var_74 = var_0.func_api(var_1, var_72, var_73, var_14, has_self=var_15, cls_method=var_15)
    var_75 = 'return_func'
    var_76 = []
    var_77 = []
    var_78 = []
    var_79 = []
    var_80 = []
    var_81 = module_1.arguments(*var_77)
    var_82 = []
    var_83 = []
    var_84 = 'bool'
    var_85 = module_1.Load()
    var_86 = module_1.Name()
    var_87 = module_1.FunctionDef(*var_81)
    var_88 = 'test_module.return_func'
    var_89 = var_87.args
    var_90 = var_87.returns
    var_91 = var_0.func_api(var_1, var_88, var_89, var_90, has_self=var_15, cls_method=var_15)
    var_92 = 'method_func'
    var_93 = []
    var_94 = 'self'
    var_95 = module_1.arg()
    var_96 = [var_95]
    var_97 = []
    var_98 = []
    var_99 = []
    var_100 = module_1.arguments(*var_96)
    var_101 = []
    var_102 = []
    var_103 = module_1.FunctionDef(*var_100)
    var_104 = 'test_module.method_func'
    var_105 = var_103.args
    var_106 = True
    var_107 = var_0.func_api(var_1, var_104, var_105, var_14, has_self=var_106, cls_method=var_15)
    var_108 = 'classmethod_func'
    var_109 = []
    var_110 = 'cls'
    var_111 = module_1.arg()
    var_112 = [var_111]
    var_113 = []
    var_114 = []
    var_115 = []
    var_116 = module_1.arguments(*var_112)
    var_117 = []
    var_118 = 'classmethod'
    var_119 = module_1.Load()
    var_120 = module_1.Name()
    var_121 = [var_120]
    var_122 = module_1.FunctionDef(*var_116)
    var_123 = 'test_module.classmethod_func'
    var_124 = var_122.args
    var_125 = var_0.func_api(var_1, var_123, var_124, var_14, has_self=var_106, cls_method=var_106)



# Parsed testcases at query #20
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_module'
    var_3 = '\nfrom typing import List\n\ndef test_function(arg1: int, arg2: str) -> bool:\n    \'\'\'Test function docstring.\'\'\'\n    return True\n\nclass TestClass:\n    \'\'\'Test class docstring.\'\'\'\n\n    def method(self, arg: float) -> None:\n        \'\'\'Test method docstring.\'\'\'\n        pass\n\n    @staticmethod\n    def static_method(arg: List[int]) -> str:\n        \'\'\'Test static method docstring.\'\'\'\n        return ""\n\n    @classmethod\n    def class_method(cls, arg: dict) -> None:\n        \'\'\'Test class method docstring.\'\'\'\n        pass\n\nasync def async_function(arg: complex) -> bytes:\n    \'\'\'Test async function docstring.\'\'\'\n    return b""\n'
    var_4 = module_0.parse(var_2, var_3)



