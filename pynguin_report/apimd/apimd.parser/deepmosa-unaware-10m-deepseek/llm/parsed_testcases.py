####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_func'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.arg()
    var_11 = [var_10]
    var_12 = None
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = module_1.arguments(*var_11)
    var_17 = []
    var_18 = []
    var_19 = 'str'
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = module_1.FunctionDef(*var_16)
    var_23 = var_0.api(var_1, var_22)
    var_24 = 'async_func'
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = []
    var_30 = module_1.arguments(*var_26)
    var_31 = []
    var_32 = []
    var_33 = module_1.AsyncFunctionDef(*var_30)
    var_34 = var_0.api(var_1, var_33)
    var_35 = 'TestClass'
    var_36 = []
    var_37 = []
    var_38 = []
    var_39 = []
    var_40 = module_1.ClassDef()
    var_41 = var_0.api(var_1, var_40)
    var_42 = 'NestedClass'
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = []
    var_47 = module_1.ClassDef()
    var_48 = var_0.api(var_1, var_47, prefix=var_35)
    var_49 = 'decorated_func'
    var_50 = []
    var_51 = []
    var_52 = []
    var_53 = []
    var_54 = []
    var_55 = module_1.arguments(*var_51)
    var_56 = []
    var_57 = 'staticmethod'
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = [var_59]
    var_61 = module_1.FunctionDef(*var_55)
    var_62 = var_0.api(var_1, var_61)
    var_63 = 'ClassWithMethod'
    var_64 = []
    var_65 = []
    var_66 = []
    var_67 = []
    var_68 = module_1.ClassDef()
    var_69 = var_0.api(var_1, var_68)
    var_70 = 'instance_method'
    var_71 = []
    var_72 = 'self'
    var_73 = module_1.arg()
    var_74 = [var_73]
    var_75 = []
    var_76 = []
    var_77 = []
    var_78 = module_1.arguments(*var_74)
    var_79 = []
    var_80 = []
    var_81 = module_1.FunctionDef(*var_78)
    var_82 = var_0.api(var_1, var_81, prefix=var_63)
    var_83 = 'class_method'
    var_84 = []
    var_85 = 'cls'
    var_86 = module_1.arg()
    var_87 = [var_86]
    var_88 = []
    var_89 = []
    var_90 = []
    var_91 = module_1.arguments(*var_87)
    var_92 = []
    var_93 = 'classmethod'
    var_94 = module_1.Load()
    var_95 = module_1.Name()
    var_96 = [var_95]
    var_97 = module_1.FunctionDef(*var_91)
    var_98 = var_0.api(var_1, var_97, prefix=var_63)
    var_99 = 'func_with_doc'
    var_100 = []
    var_101 = []
    var_102 = []
    var_103 = []
    var_104 = []
    var_105 = module_1.arguments(*var_101)
    var_106 = 'Test docstring'
    var_107 = module_1.Constant()
    var_108 = module_1.Expr()
    var_109 = [var_108]
    var_110 = []
    var_111 = module_1.FunctionDef(*var_105)
    var_112 = var_0.api(var_1, var_111)
    var_113 = True
    var_114 = module_0.Parser(var_113)
    var_115 = var_114.parse(var_1, var_2)
    var_116 = var_114.api(var_1, var_22)
    var_117 = 2
    var_118 = module_0.Parser(b_level=var_117)
    var_119 = var_118.parse(var_1, var_2)
    var_120 = var_118.api(var_1, var_22)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = 'module'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    assert var_4 is True
    var_5 = module_0.Parser()
    var_6 = 'module._private'
    var_7 = set()
    var_8 = var_5.is_public(var_6)
    assert var_8 is False
    var_9 = module_0.Parser()
    var_10 = {var_1}
    var_11 = var_9.is_public(var_1)
    assert var_11 is True
    var_12 = module_0.Parser()
    var_13 = 'module.sub.name'
    var_14 = 'module.sub'
    var_15 = {var_14}
    var_16 = var_12.is_public(var_13)
    assert var_16 is True
    var_17 = module_0.Parser()
    var_18 = 'module.other'
    var_19 = {var_18}
    var_20 = var_17.is_public(var_1)
    assert var_20 is False
    var_21 = module_0.Parser()
    var_22 = 'module.child'
    var_23 = {var_22}
    var_24 = var_21.is_public(var_2)
    assert var_24 is True
    var_25 = module_0.Parser()
    var_26 = 'module.public_child'
    var_27 = set()
    var_28 = ''
    var_29 = var_25.is_public(var_2)
    assert var_29 is True
    var_30 = module_0.Parser()
    var_31 = set()
    var_32 = var_30.is_public(var_2)
    assert var_32 is False
    var_33 = module_0.Parser()
    var_34 = 'module.__magic__'
    var_35 = set()
    var_36 = var_33.is_public(var_34)
    assert var_36 is False
    var_37 = module_0.Parser()
    var_38 = 'module.PublicName'
    var_39 = set()
    var_40 = var_37.is_public(var_38)
    assert var_40 is True



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx
    var_11 = 'collections'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'abc'
    var_15 = module_1.Load()
    var_16 = module_1.Attribute()
    var_17 = var_2.visit_Attribute(var_16)
    var_18 = var_17.value
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = 'Union'
    var_22 = module_1.Load()
    var_23 = module_1.Attribute()
    var_24 = 'Optional'
    var_25 = module_1.Load()
    var_26 = module_1.Attribute()
    var_27 = var_2.visit_Attribute(var_26)
    var_28 = var_27.value
    var_29 = 'test'
    var_30 = module_1.Constant()
    var_31 = 'attr'
    var_32 = module_1.Load()
    var_33 = module_1.Attribute()
    var_34 = var_2.visit_Attribute(var_33)
    var_35 = var_34.value
    var_36 = 'test_module.typing'
    var_37 = {var_36: var_3}
    var_38 = module_0.Resolver(var_0, var_37)
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = 'Dict'
    var_42 = module_1.Load()
    var_43 = module_1.Attribute()
    var_44 = var_38.visit_Attribute(var_43)



# Parsed testcases at query #4
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = False
    var_7 = 6
    var_8 = True
    var_9 = 7
    var_10 = []
    var_11 = 8
    var_12 = 9
    var_13 = 'e'
    var_14 = None
    var_15 = module_0.arg()
    var_16 = [var_15]
    var_17 = 10
    var_18 = 11
    var_19 = []
    var_20 = module_1.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = True
    var_23 = []
    var_24 = []
    var_25 = []
    var_26 = []



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.func1'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_1.arguments(*var_4)
    var_10 = None
    var_11 = False
    var_12 = var_0.func_api(var_1, var_2, var_9, var_10, has_self=var_11, cls_method=var_11)
    var_13 = 'test_module.func2'
    var_14 = []
    var_15 = 'x'
    var_16 = 'int'
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = module_1.arg()
    var_20 = 'y'
    var_21 = 'str'
    var_22 = module_1.Load()
    var_23 = module_1.Name()
    var_24 = module_1.arg()
    var_25 = [var_19, var_24]
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = module_1.arguments(*var_25)
    var_30 = 'bool'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = var_0.func_api(var_1, var_13, var_29, var_32, has_self=var_11, cls_method=var_11)
    var_34 = 'test_module.Class.method'
    var_35 = []
    var_36 = 'self'
    var_37 = module_1.arg()
    var_38 = 'value'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.arg()
    var_42 = [var_37, var_41]
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = module_1.arguments(*var_42)
    var_47 = None
    var_48 = True
    var_49 = var_0.func_api(var_1, var_34, var_46, var_47, has_self=var_48, cls_method=var_11)
    var_50 = 'test_module.func3'
    var_51 = []
    var_52 = 'a'
    var_53 = module_1.arg()
    var_54 = 'b'
    var_55 = module_1.arg()
    var_56 = [var_53, var_55]
    var_57 = []
    var_58 = []
    var_59 = module_1.Constant()
    var_60 = 'default'
    var_61 = module_1.Constant()
    var_62 = [var_59, var_61]
    var_63 = module_1.arguments(*var_56)
    var_64 = None
    var_65 = var_0.func_api(var_1, var_50, var_63, var_64, has_self=var_11, cls_method=var_11)
    var_66 = 'test_module.func4'
    var_67 = []
    var_68 = []
    var_69 = 'args'
    var_70 = module_1.arg()
    var_71 = []
    var_72 = []
    var_73 = 'kwargs'
    var_74 = module_1.arg()
    var_75 = []
    var_76 = module_1.arguments(*var_68)
    var_77 = None
    var_78 = var_0.func_api(var_1, var_66, var_76, var_77, has_self=var_11, cls_method=var_11)
    var_79 = 'test_module.Class.class_method'
    var_80 = []
    var_81 = 'cls'
    var_82 = module_1.arg()
    var_83 = 'param'
    var_84 = module_1.Load()
    var_85 = module_1.Name()
    var_86 = module_1.arg()
    var_87 = [var_82, var_86]
    var_88 = []
    var_89 = []
    var_90 = []
    var_91 = module_1.arguments(*var_87)
    var_92 = None
    var_93 = var_0.func_api(var_1, var_79, var_91, var_92, has_self=var_48, cls_method=var_48)
    var_94 = 'test_module.func5'
    var_95 = []
    var_96 = []
    var_97 = 'key'
    var_98 = module_1.Load()
    var_99 = module_1.Name()
    var_100 = module_1.arg()
    var_101 = module_1.Load()
    var_102 = module_1.Name()
    var_103 = module_1.arg()
    var_104 = [var_100, var_103]
    var_105 = 'default_key'
    var_106 = module_1.Constant()
    var_107 = module_1.Constant()
    var_108 = [var_106, var_107]
    var_109 = []
    var_110 = module_1.arguments(*var_96)
    var_111 = None
    var_112 = var_0.func_api(var_1, var_94, var_110, var_111, has_self=var_11, cls_method=var_11)
    var_113 = 'test_module.func6'
    var_114 = 'pos1'
    var_115 = module_1.Load()
    var_116 = module_1.Name()
    var_117 = module_1.arg()
    var_118 = 'pos2'
    var_119 = module_1.Load()
    var_120 = module_1.Name()
    var_121 = module_1.arg()
    var_122 = [var_117, var_121]
    var_123 = []
    var_124 = []
    var_125 = []
    var_126 = []
    var_127 = module_1.arguments(*var_123)
    var_128 = 'None'
    var_129 = module_1.Load()
    var_130 = module_1.Name()
    var_131 = var_0.func_api(var_1, var_113, var_127, var_130, has_self=var_11, cls_method=var_11)



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx
    var_11 = {}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'collections'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = 'abc'
    var_17 = module_1.Load()
    var_18 = module_1.Attribute()
    var_19 = var_12.visit_Attribute(var_18)
    var_20 = {}
    var_21 = module_0.Resolver(var_0, var_20)
    var_22 = module_1.Load()
    var_23 = module_1.Name()
    var_24 = 'Union'
    var_25 = module_1.Load()
    var_26 = module_1.Attribute()
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = var_21.visit_Attribute(var_28)
    var_30 = {}
    var_31 = module_0.Resolver(var_0, var_30)
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = 'Dict'
    var_35 = module_1.Load()
    var_36 = module_1.Attribute()
    var_37 = var_31.visit_Attribute(var_36)
    var_38 = var_37.ctx
    var_39 = {}
    var_40 = module_0.Resolver(var_0, var_39)
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = 'Optional'
    var_44 = module_1.Load()
    var_45 = module_1.Attribute()
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = 'Callable'
    var_49 = module_1.Load()
    var_50 = module_1.Attribute()
    var_51 = var_40.visit_Attribute(var_45)
    var_52 = var_40.visit_Attribute(var_50)



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = ''
    var_10 = var_0.parse(var_1, var_9)
    var_11 = module_0.Parser()
    var_12 = 'my_var'
    var_13 = 'hello'
    var_14 = module_1.Constant()
    var_15 = 'str'
    var_16 = var_11.parse(var_1, var_9)
    var_17 = module_0.Parser()
    var_18 = 3.14
    var_19 = module_1.Constant()
    var_20 = None
    var_21 = var_17.parse(var_1, var_9)
    var_22 = 'test_module.my_var'
    var_23 = module_0.Parser()
    var_24 = '__all__'
    var_25 = 'func1'
    var_26 = module_1.Constant()
    var_27 = 'ClassA'
    var_28 = module_1.Constant()
    var_29 = [var_26, var_28]
    var_30 = module_1.Load()
    var_31 = module_1.Tuple()
    var_32 = var_23.parse(var_1, var_9)
    var_33 = module_0.Parser()
    var_34 = 'CONSTANT'
    var_35 = module_1.Constant()
    var_36 = [var_35]
    var_37 = module_1.Load()
    var_38 = module_1.List()
    var_39 = var_33.parse(var_1, var_9)
    var_40 = module_0.Parser()
    var_41 = 'lowercase'
    var_42 = 100
    var_43 = module_1.Constant()
    var_44 = var_40.parse(var_1, var_9)
    var_45 = module_0.Parser()
    var_46 = 'EXISTING'
    var_47 = 'new_value'
    var_48 = module_1.Constant()
    var_49 = var_45.parse(var_1, var_9)
    var_50 = module_0.Parser()
    var_51 = 'no_value'
    var_52 = 'Optional[int]'
    var_53 = module_1.Load()
    var_54 = module_1.Name()
    var_55 = var_50.parse(var_1, var_9)
    var_56 = module_0.Parser()
    var_57 = 'a'
    var_58 = 'b'
    var_59 = module_1.Constant()
    var_60 = var_56.parse(var_1, var_9)
    var_61 = module_0.Parser()
    var_62 = 'module'
    var_63 = module_1.Load()
    var_64 = module_1.Name()
    var_65 = 'attr'
    var_66 = 2
    var_67 = module_1.Constant()
    var_68 = var_61.parse(var_1, var_9)



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = 'y'
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.arg()
    var_11 = 'return'
    var_12 = 'bool'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = module_1.arg()
    var_16 = [var_5, var_10, var_15]
    var_17 = 'test_module'
    var_18 = False
    var_19 = var_0.func_ann(var_17, var_16, has_self=var_18, cls_method=var_18)
    var_20 = list(var_19)
    var_21 = 'self'
    var_22 = None
    var_23 = module_1.arg()
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = module_1.arg()
    var_27 = 'None'
    var_28 = module_1.Load()
    var_29 = module_1.Name()
    var_30 = module_1.arg()
    var_31 = [var_23, var_26, var_30]
    var_32 = True
    var_33 = var_0.func_ann(var_17, var_31, has_self=var_32, cls_method=var_18)
    var_34 = list(var_33)
    var_35 = 'cls'
    var_36 = 'type'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = module_1.arg()
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = module_1.arg()
    var_43 = module_1.Load()
    var_44 = module_1.Name()
    var_45 = module_1.arg()
    var_46 = [var_39, var_42, var_45]
    var_47 = var_0.func_ann(var_17, var_46, has_self=var_32, cls_method=var_32)
    var_48 = list(var_47)
    var_49 = 'MyClass'
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = module_1.arg()
    var_53 = module_1.Load()
    var_54 = module_1.Name()
    var_55 = module_1.arg()
    var_56 = module_1.arg()
    var_57 = [var_52, var_55, var_56]
    var_58 = var_0.func_ann(var_17, var_57, has_self=var_32, cls_method=var_18)
    var_59 = list(var_58)
    var_60 = module_1.Load()
    var_61 = module_1.Name()
    var_62 = module_1.arg()
    var_63 = '*'
    var_64 = module_1.arg()
    var_65 = module_1.Load()
    var_66 = module_1.Name()
    var_67 = module_1.arg()
    var_68 = module_1.arg()
    var_69 = [var_62, var_64, var_67, var_68]
    var_70 = var_0.func_ann(var_17, var_69, has_self=var_18, cls_method=var_18)
    var_71 = list(var_70)
    var_72 = module_1.arg()
    var_73 = module_1.arg()
    var_74 = [var_72, var_73]
    var_75 = var_0.func_ann(var_17, var_74, has_self=var_18, cls_method=var_18)
    var_76 = list(var_75)
    var_77 = 'test_module.int'
    var_78 = 'builtins.int'
    var_79 = module_1.Load()
    var_80 = module_1.Name()
    var_81 = module_1.arg()
    var_82 = module_1.Load()
    var_83 = module_1.Name()
    var_84 = module_1.arg()
    var_85 = [var_81, var_84]
    var_86 = var_0.func_ann(var_17, var_85, has_self=var_18, cls_method=var_18)
    var_87 = list(var_86)
    var_88 = module_1.Load()
    var_89 = module_1.Name()
    var_90 = module_1.arg()
    var_91 = 'other'
    var_92 = 'Self'
    var_93 = module_1.Load()
    var_94 = module_1.Name()
    var_95 = module_1.arg()
    var_96 = module_1.arg()
    var_97 = [var_90, var_95, var_96]
    var_98 = var_0.func_ann(var_17, var_97, has_self=var_32, cls_method=var_18)
    var_99 = list(var_98)



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module2'
    var_5 = 'import os\ndef bar(): pass'
    var_6 = var_0.parse(var_4, var_5)
    var_7 = 'test_module3'
    var_8 = 'class MyClass:\n    def method(self): pass'
    var_9 = var_0.parse(var_7, var_8)
    var_10 = 'test_module4'
    var_11 = 'async def async_func(): pass'
    var_12 = var_0.parse(var_10, var_11)
    var_13 = 'test_module5'
    var_14 = '"""Module docstring."""\ndef func(): pass'
    var_15 = var_0.parse(var_13, var_14)
    var_16 = 'test_module6'
    var_17 = "CONSTANT = 42\n__all__ = ['foo']"
    var_18 = var_0.parse(var_16, var_17)
    var_19 = 'test_module7'
    var_20 = 'from typing import List\nVector = List[float]'
    var_21 = var_0.parse(var_19, var_20)
    var_22 = 'parent.child'
    var_23 = 'def nested(): pass'
    var_24 = var_0.parse(var_22, var_23)
    var_25 = 'test_module8'
    var_26 = '@staticmethod\ndef static_method(): pass'
    var_27 = var_0.parse(var_25, var_26)
    var_28 = 'empty_module'
    var_29 = ''
    var_30 = var_0.parse(var_28, var_29)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.func1'
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_3)
    var_9 = None
    var_10 = 'test_module'
    var_11 = False
    var_12 = var_0.func_api(var_10, var_1, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_13 = 'test_module.func2'
    var_14 = 'x'
    var_15 = 'int'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = module_1.arg()
    var_19 = [var_18]
    var_20 = 'y'
    var_21 = 'str'
    var_22 = module_1.Load()
    var_23 = module_1.Name()
    var_24 = module_1.arg()
    var_25 = [var_24]
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = module_1.arguments(*var_25)
    var_30 = 'bool'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = var_0.func_api(var_10, var_13, var_29, var_32, has_self=var_11, cls_method=var_11)
    var_34 = 'test_module.func3'
    var_35 = []
    var_36 = 'a'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = module_1.arg()
    var_40 = 'b'
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = module_1.arg()
    var_44 = [var_39, var_43]
    var_45 = []
    var_46 = []
    var_47 = 1
    var_48 = module_1.Constant()
    var_49 = 'default'
    var_50 = module_1.Constant()
    var_51 = [var_48, var_50]
    var_52 = module_1.arguments(*var_44)
    var_53 = 'None'
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = var_0.func_api(var_10, var_34, var_52, var_55, has_self=var_11, cls_method=var_11)
    var_57 = 'test_module.func4'
    var_58 = []
    var_59 = []
    var_60 = 'args'
    var_61 = 'Any'
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = module_1.arg()
    var_65 = []
    var_66 = []
    var_67 = 'kwargs'
    var_68 = module_1.Load()
    var_69 = module_1.Name()
    var_70 = module_1.arg()
    var_71 = []
    var_72 = module_1.arguments(*var_59)
    var_73 = module_1.Load()
    var_74 = module_1.Name()
    var_75 = var_0.func_api(var_10, var_57, var_72, var_74, has_self=var_11, cls_method=var_11)
    var_76 = 'test_module.Class.method'
    var_77 = []
    var_78 = 'self'
    var_79 = module_1.arg()
    var_80 = module_1.Load()
    var_81 = module_1.Name()
    var_82 = module_1.arg()
    var_83 = [var_79, var_82]
    var_84 = []
    var_85 = []
    var_86 = []
    var_87 = module_1.arguments(*var_83)
    var_88 = module_1.Load()
    var_89 = module_1.Name()
    var_90 = True
    var_91 = var_0.func_api(var_10, var_76, var_87, var_89, has_self=var_90, cls_method=var_11)
    var_92 = 'test_module.Class.class_method'
    var_93 = []
    var_94 = 'cls'
    var_95 = 'type'
    var_96 = module_1.Load()
    var_97 = module_1.Name()
    var_98 = module_1.arg()
    var_99 = module_1.Load()
    var_100 = module_1.Name()
    var_101 = module_1.arg()
    var_102 = [var_98, var_101]
    var_103 = []
    var_104 = []
    var_105 = []
    var_106 = module_1.arguments(*var_102)
    var_107 = module_1.Load()
    var_108 = module_1.Name()
    var_109 = True
    var_110 = True
    var_111 = var_0.func_api(var_10, var_92, var_106, var_108, has_self=var_109, cls_method=var_110)
    var_112 = 'test_module.func5'
    var_113 = []
    var_114 = []
    var_115 = module_1.Load()
    var_116 = module_1.Name()
    var_117 = module_1.arg()
    var_118 = module_1.Load()
    var_119 = module_1.Name()
    var_120 = module_1.arg()
    var_121 = [var_117, var_120]
    var_122 = module_1.Constant()
    var_123 = 'test'
    var_124 = module_1.Constant()
    var_125 = [var_122, var_124]
    var_126 = []
    var_127 = module_1.arguments(*var_114)
    var_128 = None
    var_129 = var_0.func_api(var_10, var_112, var_127, var_128, has_self=var_11, cls_method=var_11)
    var_130 = 'test_module.func6'
    var_131 = module_1.Load()
    var_132 = module_1.Name()
    var_133 = module_1.arg()
    var_134 = module_1.Load()
    var_135 = module_1.Name()
    var_136 = module_1.arg()
    var_137 = [var_133, var_136]
    var_138 = 'c'
    var_139 = module_1.Load()
    var_140 = module_1.Name()
    var_141 = module_1.arg()
    var_142 = [var_141]
    var_143 = []
    var_144 = []
    var_145 = []
    var_146 = module_1.arguments(*var_142)
    var_147 = None
    var_148 = var_0.func_api(var_10, var_130, var_146, var_147, has_self=var_11, cls_method=var_11)



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = 'collections'
    var_8 = module_1.Load()
    var_9 = 'abc'
    var_10 = module_1.Load()
    var_11 = 'test'
    var_12 = 'attr'
    var_13 = module_1.Load()
    var_14 = 'test_module.typing'
    var_15 = {var_14: var_3}
    var_16 = module_0.Resolver(var_0, var_15)
    var_17 = module_1.Load()
    var_18 = 'Union'
    var_19 = module_1.Load()
    var_20 = 'other_module'
    var_21 = module_1.Load()
    var_22 = 'Something'
    var_23 = module_1.Load()



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 0
    var_4 = 'Union[int, str]'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = 'Optional[int]'
    var_9 = module_1.parse(var_8)
    var_10 = var_9.body[var_3]
    var_11 = var_10.value
    var_12 = 0
    var_13 = 'List[int]'
    var_14 = module_1.parse(var_13)
    var_15 = var_14.body[var_12]
    var_16 = var_15.value
    var_17 = 'Custom[int]'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_14]
    var_20 = var_19.value
    var_21 = 'Union[int]'
    var_22 = module_1.parse(var_21)
    var_23 = var_22.body[var_14]
    var_24 = var_23.value
    var_25 = 'test_module.Union'
    var_26 = 'typing.Union'
    var_27 = {var_25: var_26}
    var_28 = module_0.Resolver(var_12, var_27)
    var_29 = module_1.parse(var_15)
    var_30 = var_29.body[var_14]
    var_31 = var_30.value
    var_32 = module_1.parse(var_21)
    var_33 = var_32.body[var_14]
    var_34 = var_33.value
    var_35 = 'int'
    var_36 = module_1.Load()



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = []
    var_7 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_8 = 'BaseClass'
    var_9 = module_1.Load()
    var_10 = 'AnotherBase'
    var_11 = module_1.Load()
    var_12 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_13 = 'enum'
    var_14 = module_1.Load()
    var_15 = 'Enum'
    var_16 = module_1.Load()
    var_17 = 'RED'
    var_18 = 1
    var_19 = None
    var_20 = 'GREEN'
    var_21 = 2
    var_22 = 'BLUE'
    var_23 = 3
    var_24 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_25 = []
    var_26 = 'public_attr'
    var_27 = 'int'
    var_28 = module_1.Load()
    var_29 = '_private_attr'
    var_30 = 'str'
    var_31 = module_1.Load()
    var_32 = 'another_public'
    var_33 = 'value'
    var_34 = '_another_private'
    var_35 = 42
    var_36 = var_0.class_api(var_1, var_4, var_25, var_6)
    var_37 = 'with_type'
    var_38 = var_0.class_api(var_1, var_4, var_25, var_6)
    var_39 = 'attr1'
    var_40 = module_1.Load()
    var_41 = 'attr2'
    var_42 = module_1.Load()
    var_43 = var_0.class_api(var_1, var_4, var_25, var_6)
    var_44 = module_1.Load()
    var_45 = module_1.Load()
    var_46 = 'VAL1'
    var_47 = 'VAL2'
    var_48 = var_0.class_api(var_1, var_4, var_25, var_6)
    var_49 = module_1.Load()
    var_50 = module_1.Load()
    var_51 = 'ENUM_VAL'
    var_52 = 'regular_attr'
    var_53 = module_1.Load()
    var_54 = var_0.class_api(var_1, var_4, var_25, var_6)
    var_55 = 'CustomEnum'
    var_56 = module_1.Load()
    var_57 = 'OPTION'
    var_58 = var_0.class_api(var_1, var_4, var_25, var_6)



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx
    var_11 = 'collections'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'abc'
    var_15 = module_1.Load()
    var_16 = module_1.Attribute()
    var_17 = var_2.visit_Attribute(var_16)
    var_18 = var_17.ctx
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = 'extensions'
    var_22 = module_1.Load()
    var_23 = module_1.Attribute()
    var_24 = 'Protocol'
    var_25 = module_1.Load()
    var_26 = module_1.Attribute()
    var_27 = var_2.visit_Attribute(var_26)
    var_28 = var_27.ctx
    var_29 = 'get_module'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = []
    var_33 = []
    var_34 = module_1.Call(*var_32)
    var_35 = module_1.Load()
    var_36 = module_1.Attribute()
    var_37 = var_2.visit_Attribute(var_36)
    var_38 = var_37.value
    var_39 = var_37.ctx



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.func1'
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_3)
    var_9 = None
    var_10 = 'test_module'
    var_11 = False
    var_12 = var_0.func_api(var_10, var_1, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_13 = 'test_module.func2'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = 'z'
    var_17 = []
    var_18 = []
    var_19 = 1
    var_20 = 2
    var_21 = 'int'
    var_22 = module_1.Load()
    var_23 = var_0.func_api(var_10, var_13, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_24 = 'test_module.Class.method'
    var_25 = []
    var_26 = 'self'
    var_27 = []
    var_28 = []
    var_29 = 10
    var_30 = 'str'
    var_31 = module_1.Load()
    var_32 = True
    var_33 = var_0.func_api(var_10, var_24, var_8, var_9, has_self=var_32, cls_method=var_11)
    var_34 = 'test_module.Class.class_method'
    var_35 = []
    var_36 = 'cls'
    var_37 = 'Type'
    var_38 = module_1.Load()
    var_39 = []
    var_40 = []
    var_41 = []
    var_42 = 'None'
    var_43 = module_1.Load()
    var_44 = True
    var_45 = True
    var_46 = var_0.func_api(var_10, var_34, var_8, var_9, has_self=var_44, cls_method=var_45)
    var_47 = 'test_module.func3'
    var_48 = []
    var_49 = []
    var_50 = 'args'
    var_51 = 'kwargs'
    var_52 = []
    var_53 = var_0.func_api(var_10, var_47, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_54 = 'test_module.func4'
    var_55 = []
    var_56 = module_1.Load()
    var_57 = module_1.Load()
    var_58 = []
    var_59 = []
    var_60 = []
    var_61 = 'list'
    var_62 = module_1.Load()
    var_63 = module_1.Load()
    var_64 = module_1.Load()
    var_65 = var_0.func_api(var_10, var_54, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_66 = 'test_module.func5'
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = module_1.Load()
    var_71 = []
    var_72 = []
    var_73 = None
    var_74 = var_0.func_api(var_10, var_66, var_8, var_73, has_self=var_11, cls_method=var_11)
    var_75 = 'test_module.func6'
    var_76 = []
    var_77 = []
    var_78 = []
    var_79 = None
    var_80 = var_0.func_api(var_10, var_75, var_8, var_79, has_self=var_11, cls_method=var_11)



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.TestClass'
    var_5 = []
    var_6 = []
    var_7 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_8 = 'test_module.TestClass2'
    var_9 = 'BaseClass'
    var_10 = module_1.Load()
    var_11 = 'AnotherBase'
    var_12 = module_1.Load()
    var_13 = []
    var_14 = var_0.class_api(var_1, var_8, var_5, var_13)
    var_15 = 'test_module.TestEnum'
    var_16 = 'enum'
    var_17 = module_1.Load()
    var_18 = 'Enum'
    var_19 = module_1.Load()
    var_20 = 'RED'
    var_21 = 1
    var_22 = None
    var_23 = 'GREEN'
    var_24 = 2
    var_25 = 'BLUE'
    var_26 = 3
    var_27 = var_0.class_api(var_1, var_15, var_5, var_13)
    var_28 = 'test_module.TestClassWithMembers'
    var_29 = []
    var_30 = 'public_attr'
    var_31 = 'int'
    var_32 = module_1.Load()
    var_33 = '_private_attr'
    var_34 = 'str'
    var_35 = module_1.Load()
    var_36 = 'protected_attr'
    var_37 = 'float'
    var_38 = module_1.Load()
    var_39 = var_0.class_api(var_1, var_28, var_29, var_13)
    var_40 = 'test_module.TestClassWithAssign'
    var_41 = []
    var_42 = 'attr1'
    var_43 = 'value'
    var_44 = '# type: str'
    var_45 = 'attr2'
    var_46 = 123
    var_47 = '# type: int'
    var_48 = var_0.class_api(var_1, var_40, var_41, var_13)
    var_49 = 'test_module.TestClassWithDelete'
    var_50 = module_1.Load()
    var_51 = module_1.Load()
    var_52 = 'ITEM1'
    var_53 = 'ITEM2'
    var_54 = module_1.Load()
    var_55 = var_0.class_api(var_1, var_49, var_41, var_13)
    var_56 = 'test_module.TestMixed'
    var_57 = 'object'
    var_58 = module_1.Load()
    var_59 = 'CONSTANT'
    var_60 = 100
    var_61 = 'regular_attr'
    var_62 = module_1.Load()
    var_63 = var_0.class_api(var_1, var_56, var_41, var_13)



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not valid python code'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'List[int]'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = 'typing.List'
    var_13 = module_1.Constant()
    var_14 = var_2.visit_Constant(var_13)
    var_15 = var_14.value
    var_16 = ''
    var_17 = module_1.Constant()
    var_18 = var_2.visit_Constant(var_17)
    var_19 = 'Union[str, int]'
    var_20 = module_1.Constant()
    var_21 = var_2.visit_Constant(var_20)
    var_22 = var_21.left
    var_23 = var_21.right
    var_24 = var_21.op



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing.List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)
    var_12 = var_11.value
    var_13 = 'typing.Union'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = 'str'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = [var_17, var_20]
    var_22 = module_1.Load()
    var_23 = module_1.Tuple()
    var_24 = module_1.Load()
    var_25 = module_1.Subscript()
    var_26 = var_2.visit_Subscript(var_25)
    var_27 = var_26.op
    var_28 = 'typing.Optional'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = module_1.Load()
    var_34 = module_1.Subscript()
    var_35 = var_2.visit_Subscript(var_34)
    var_36 = var_35.op
    var_37 = var_35.right
    var_38 = 'CustomType'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = module_1.Load()
    var_44 = module_1.Subscript()
    var_45 = var_2.visit_Subscript(var_44)
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = module_1.Load()
    var_51 = module_1.Subscript()
    var_52 = var_2.visit_Subscript(var_51)
    var_53 = 'test_module.typing.List'
    var_54 = {var_53: var_3}
    var_55 = module_0.Resolver(var_0, var_54)
    var_56 = var_55.visit_Subscript(var_10)



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx
    var_11 = 'collections'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'abc'
    var_15 = module_1.Load()
    var_16 = module_1.Attribute()
    var_17 = var_2.visit_Attribute(var_16)
    var_18 = var_17.value
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = 'extensions'
    var_22 = module_1.Load()
    var_23 = module_1.Attribute()
    var_24 = 'Protocol'
    var_25 = module_1.Load()
    var_26 = module_1.Attribute()
    var_27 = var_2.visit_Attribute(var_26)
    var_28 = var_27.value



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not a valid name syntax'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'test_module.MyClass'
    var_10 = 'MyClass'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Constant()
    var_14 = var_12.visit_Constant(var_13)
    var_15 = var_14.ctx
    var_16 = 'test_module.submodule.Class'
    var_17 = 'submodule.Class'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = module_1.Constant()
    var_21 = var_19.visit_Constant(var_20)
    var_22 = 'test_module.OldName'
    var_23 = 'NewName'
    var_24 = {var_22: var_23}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = 'OldName'
    var_27 = module_1.Constant()
    var_28 = var_25.visit_Constant(var_27)
    var_29 = {}
    var_30 = module_0.Resolver(var_0, var_29)
    var_31 = 'List[int]'
    var_32 = module_1.Constant()
    var_33 = var_30.visit_Constant(var_32)
    var_34 = var_33.value
    var_35 = {}
    var_36 = module_0.Resolver(var_0, var_35)
    var_37 = "TypeVar('T')"
    var_38 = module_1.Constant()
    var_39 = var_36.visit_Constant(var_38)
    var_40 = var_39.func
    var_41 = ''
    var_42 = module_1.Constant()
    var_43 = var_36.visit_Constant(var_42)
    var_44 = 'my-name'
    var_45 = module_1.Constant()
    var_46 = var_36.visit_Constant(var_45)
    var_47 = {}
    var_48 = 'SelfType'
    var_49 = module_0.Resolver(var_0, var_47, var_48)
    var_50 = module_1.Constant()
    var_51 = var_49.visit_Constant(var_50)



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = var_9.ctx
    var_11 = 'collections'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'abc'
    var_15 = module_1.Load()
    var_16 = module_1.Attribute()
    var_17 = var_2.visit_Attribute(var_16)
    var_18 = var_17.value
    var_19 = var_17.ctx
    var_20 = 'test'
    var_21 = module_1.Constant()
    var_22 = 'attr'
    var_23 = module_1.Load()
    var_24 = module_1.Attribute()
    var_25 = var_2.visit_Attribute(var_24)
    var_26 = var_25.value
    var_27 = var_25.ctx
    var_28 = 'test_module.typing'
    var_29 = {var_28: var_3}
    var_30 = module_0.Resolver(var_0, var_29)
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = 'Dict'
    var_34 = module_1.Load()
    var_35 = module_1.Attribute()
    var_36 = var_30.visit_Attribute(var_35)
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = 'Optional'
    var_40 = module_1.Load()
    var_41 = module_1.Attribute()
    var_42 = var_30.visit_Attribute(var_41)



# Parsed testcases at query #22
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = False
    var_7 = 6
    var_8 = True
    var_9 = 7
    var_10 = []
    var_11 = 8
    var_12 = 'e'
    var_13 = None
    var_14 = module_0.arg()
    var_15 = [var_14]
    var_16 = 9
    var_17 = 10
    var_18 = 11
    var_19 = 12
    var_20 = []
    var_21 = []
    var_22 = True
    var_23 = 13
    var_24 = []
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = 14
    var_29 = 15
    var_30 = []
    var_31 = module_1.walk_body(var_30)
    var_32 = list(var_31)
    var_33 = True
    var_34 = []
    var_35 = []



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\ndef simple_func():\n    """Simple function."""\n    pass\n\nasync def async_func():\n    """Async function."""\n    pass\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = module_0.Parser()
    var_5 = 'nested'
    var_6 = '\nclass Outer:\n    class Inner:\n        """Inner class."""\n        pass\n    \n    def method(self):\n        """Method."""\n        pass\n'
    var_7 = var_4.parse(var_5, var_6)
    var_8 = module_0.Parser()
    var_9 = 'decorated'
    var_10 = '\n@decorator1\n@decorator2\ndef decorated_func():\n    pass\n'
    var_11 = var_8.parse(var_9, var_10)
    var_12 = True
    var_13 = module_0.Parser(var_12)
    var_14 = 'linked'
    var_15 = '\ndef linked_func():\n    pass\n'
    var_16 = var_13.parse(var_14, var_15)
    var_17 = False
    var_18 = module_0.Parser(var_17)
    var_19 = 'unlinked'
    var_20 = '\ndef unlinked_func():\n    pass\n'
    var_21 = var_18.parse(var_19, var_20)



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = var_6.ctx
    var_8 = 'module.name'
    var_9 = 'other_module.OtherName'
    var_10 = {var_8: var_9}
    var_11 = ''
    var_12 = module_0.Resolver(var_0, var_10, var_11)
    var_13 = 'name'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = var_12.visit_Name(var_15)
    var_17 = var_16.ctx
    var_18 = {var_8: var_8}
    var_19 = module_0.Resolver(var_0, var_18, var_11)
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = var_19.visit_Name(var_21)
    var_23 = var_22.ctx
    var_24 = {}
    var_25 = module_0.Resolver(var_0, var_24, var_11)
    var_26 = 'unknown'
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = var_25.visit_Name(var_28)
    var_30 = 'module.T'
    var_31 = 'typing.TypeVar'
    var_32 = {var_30: var_31}
    var_33 = module_0.Resolver(var_0, var_32, var_11)
    var_34 = 'T'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = var_33.visit_Name(var_36)
    var_38 = var_37.ctx
    var_39 = 'other_module.submodule.ClassName'
    var_40 = {var_8: var_39}
    var_41 = module_0.Resolver(var_0, var_40, var_11)
    var_42 = module_1.Load()
    var_43 = module_1.Name()
    var_44 = var_41.visit_Name(var_43)
    var_45 = var_44.value
    var_46 = var_44.value.value
    var_47 = 'some_func()'
    var_48 = {var_8: var_47}
    var_49 = module_0.Resolver(var_0, var_48, var_11)
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = var_49.visit_Name(var_51)
    var_53 = var_52.func
    var_54 = 'other_module.name'
    var_55 = 'replacement'
    var_56 = {var_54: var_55}
    var_57 = 'other_module'
    var_58 = module_0.Resolver(var_57, var_56, var_11)
    var_59 = module_1.Load()
    var_60 = module_1.Name()
    var_61 = var_58.visit_Name(var_60)
    var_62 = var_61.ctx



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = 'y'
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.arg()
    var_11 = 'return'
    var_12 = 'bool'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = module_1.arg()
    var_16 = [var_5, var_10, var_15]
    var_17 = 'test_module'
    var_18 = False
    var_19 = var_0.func_ann(var_17, var_16, has_self=var_18, cls_method=var_18)
    var_20 = list(var_19)
    var_21 = 'self'
    var_22 = 'Self'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = module_1.arg()
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.arg()
    var_29 = 'None'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_1.arg()
    var_33 = [var_25, var_28, var_32]
    var_34 = True
    var_35 = var_0.func_ann(var_17, var_33, has_self=var_34, cls_method=var_18)
    var_36 = list(var_35)
    var_37 = 'cls'
    var_38 = 'type'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = module_1.Load()
    var_44 = module_1.Subscript()
    var_45 = module_1.arg()
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = module_1.arg()
    var_49 = module_1.Load()
    var_50 = module_1.Name()
    var_51 = module_1.arg()
    var_52 = [var_45, var_48, var_51]
    var_53 = var_0.func_ann(var_17, var_52, has_self=var_34, cls_method=var_34)
    var_54 = list(var_53)
    var_55 = module_1.Load()
    var_56 = module_1.Name()
    var_57 = module_1.arg()
    var_58 = '*'
    var_59 = None
    var_60 = module_1.arg()
    var_61 = module_1.Load()
    var_62 = module_1.Name()
    var_63 = module_1.arg()
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = module_1.arg()
    var_67 = [var_57, var_60, var_63, var_66]
    var_68 = var_0.func_ann(var_17, var_67, has_self=var_18, cls_method=var_18)
    var_69 = list(var_68)
    var_70 = module_1.arg()
    var_71 = module_1.arg()
    var_72 = module_1.arg()
    var_73 = [var_70, var_71, var_72]
    var_74 = var_0.func_ann(var_17, var_73, has_self=var_18, cls_method=var_18)
    var_75 = list(var_74)
    var_76 = 'CustomType'
    var_77 = module_1.Load()
    var_78 = module_1.Name()
    var_79 = module_1.arg()
    var_80 = module_1.Load()
    var_81 = module_1.Name()
    var_82 = module_1.arg()
    var_83 = [var_79, var_82]
    var_84 = var_0.func_ann(var_17, var_83, has_self=var_18, cls_method=var_18)
    var_85 = list(var_84)
    var_86 = 'MyClass'
    var_87 = module_1.Load()
    var_88 = module_1.Name()
    var_89 = module_1.arg()
    var_90 = module_1.Load()
    var_91 = module_1.Name()
    var_92 = module_1.arg()
    var_93 = module_1.Load()
    var_94 = module_1.Name()
    var_95 = module_1.arg()
    var_96 = [var_89, var_92, var_95]
    var_97 = var_0.func_ann(var_17, var_96, has_self=var_34, cls_method=var_18)
    var_98 = list(var_97)
    var_99 = module_1.Load()
    var_100 = module_1.Name()
    var_101 = module_1.arg()
    var_102 = module_1.Load()
    var_103 = module_1.Name()
    var_104 = module_1.arg()
    var_105 = '*args'
    var_106 = 'Any'
    var_107 = module_1.Load()
    var_108 = module_1.Name()
    var_109 = module_1.arg()
    var_110 = module_1.Load()
    var_111 = module_1.Name()
    var_112 = module_1.arg()
    var_113 = '**kwargs'
    var_114 = 'dict'
    var_115 = module_1.Load()
    var_116 = module_1.Name()
    var_117 = module_1.arg()
    var_118 = module_1.Load()
    var_119 = module_1.Name()
    var_120 = module_1.arg()
    var_121 = [var_101, var_104, var_109, var_112, var_117, var_120]
    var_122 = var_0.func_ann(var_17, var_121, has_self=var_34, cls_method=var_18)
    var_123 = list(var_122)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class EmptyClass:\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.EmptyClass'
    var_5 = module_0.Parser()
    var_6 = 'class ChildClass(Base1, Base2):\n    pass'
    var_7 = var_5.parse(var_1, var_6)
    var_8 = 'test_module.ChildClass'
    var_9 = module_0.Parser()
    var_10 = '\nclass MyClass:\n    x: int\n    y: str\n    _private: float\n'
    var_11 = var_9.parse(var_1, var_10)
    var_12 = 'test_module.MyClass'
    var_13 = module_0.Parser()
    var_14 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'
    var_15 = var_13.parse(var_1, var_14)
    var_16 = 'test_module.MyClass'
    var_17 = module_0.Parser()
    var_18 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_19 = var_17.parse(var_1, var_18)
    var_20 = 'test_module.Color'
    var_21 = module_0.Parser()
    var_22 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_23 = var_21.parse(var_1, var_22)
    var_24 = 'test_module.MyClass'
    var_25 = module_0.Parser()
    var_26 = '\nclass MyClass:\n    x = 42  # type: int\n    y = "hello"  # type: str\n'
    var_27 = var_25.parse(var_1, var_26)
    var_28 = 'test_module.MyClass'
    var_29 = module_0.Parser()
    var_30 = '\nfrom enum import IntEnum\nclass MyEnum(IntEnum, BaseClass):\n    VALUE1 = 1\n    VALUE2 = 2\n'
    var_31 = var_29.parse(var_1, var_30)
    var_32 = 'test_module.MyEnum'
    var_33 = module_0.Parser()
    var_34 = '\nclass PrivateClass:\n    _private1: int\n    _private2: str\n    __very_private = True\n'
    var_35 = var_33.parse(var_1, var_34)
    var_36 = 'test_module.PrivateClass'



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPE_ALIAS'
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'list'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1
    var_10 = 'CONSTANT'
    var_11 = 42
    var_12 = module_1.Constant()
    var_13 = 'int'
    var_14 = 'ANOTHER'
    var_15 = module_1.Constant()
    var_16 = 2
    var_17 = module_1.Constant()
    var_18 = [var_15, var_17]
    var_19 = module_1.Load()
    var_20 = module_1.List()
    var_21 = None
    var_22 = '__all__'
    var_23 = 'func1'
    var_24 = module_1.Constant()
    var_25 = 'Class1'
    var_26 = module_1.Constant()
    var_27 = [var_24, var_26]
    var_28 = module_1.Load()
    var_29 = module_1.Tuple()
    var_30 = 'var1'
    var_31 = module_1.Constant()
    var_32 = 'var2'
    var_33 = module_1.Constant()
    var_34 = [var_31, var_33]
    var_35 = module_1.Load()
    var_36 = module_1.List()
    var_37 = 'lowercase'
    var_38 = 'str'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = 'test'
    var_42 = module_1.Constant()
    var_43 = 'a'
    var_44 = 'b'
    var_45 = module_1.Constant()
    var_46 = 'obj'
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = 'attr'
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = 5
    var_53 = module_1.Constant()
    var_54 = 0



# Parsed testcases at query #4
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = False
    var_7 = 6
    var_8 = True
    var_9 = 7
    var_10 = []
    var_11 = 8
    var_12 = 'e'
    var_13 = None
    var_14 = module_0.arg()
    var_15 = [var_14]
    var_16 = 9
    var_17 = 10
    var_18 = 11
    var_19 = 12
    var_20 = []
    var_21 = []
    var_22 = True
    var_23 = 13
    var_24 = []
    var_25 = 14
    var_26 = []
    var_27 = 15
    var_28 = 16
    var_29 = 17
    var_30 = []
    var_31 = module_1.walk_body(var_30)
    var_32 = list(var_31)
    var_33 = True
    var_34 = []
    var_35 = []



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = var_6.ctx
    var_8 = 'module.name'
    var_9 = 'alias_name'
    var_10 = {var_8: var_9}
    var_11 = ''
    var_12 = module_0.Resolver(var_0, var_10, var_11)
    var_13 = 'name'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = var_12.visit_Name(var_15)
    var_17 = var_16.ctx
    var_18 = {var_8: var_8}
    var_19 = module_0.Resolver(var_0, var_18, var_11)
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = var_19.visit_Name(var_21)
    var_23 = var_22.ctx
    var_24 = {}
    var_25 = module_0.Resolver(var_0, var_24, var_11)
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = var_25.visit_Name(var_27)
    var_29 = 'module.TypeVar'
    var_30 = 'typing.TypeVar'
    var_31 = {var_29: var_30}
    var_32 = module_0.Resolver(var_0, var_31, var_11)
    var_33 = 'TypeVar'
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = var_32.visit_Name(var_35)
    var_37 = var_36.ctx
    var_38 = 'List[int]'
    var_39 = {var_8: var_38}
    var_40 = module_0.Resolver(var_0, var_39, var_11)
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = var_40.visit_Name(var_42)
    var_44 = var_43.value
    var_45 = var_43.slice
    var_46 = 'module.outer'
    var_47 = 'module.inner'
    var_48 = 'inner'
    var_49 = 'final'
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = module_0.Resolver(var_0, var_50, var_11)
    var_52 = 'outer'
    var_53 = module_1.Load()
    var_54 = module_1.Name()
    var_55 = var_51.visit_Name(var_54)
    var_56 = var_55.ctx



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not valid python code'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'List[int]'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = 'SomeType'
    var_13 = module_1.Constant()
    var_14 = var_2.visit_Constant(var_13)
    var_15 = 'typing.List'
    var_16 = module_1.Constant()
    var_17 = var_2.visit_Constant(var_16)
    var_18 = 'test_module.SomeType'
    var_19 = 'typing.List[int]'
    var_20 = module_1.Constant()
    var_21 = var_2.visit_Constant(var_20)
    var_22 = var_21.value
    var_23 = 'SelfType'
    var_24 = module_1.Constant()
    var_25 = var_2.visit_Constant(var_24)



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = module_1.arg()
    var_7 = 'y'
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.arg()
    var_12 = 'return'
    var_13 = 'bool'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = module_1.arg()
    var_17 = [var_6, var_11, var_16]
    var_18 = False
    var_19 = var_0.func_ann(var_1, var_17, has_self=var_18, cls_method=var_18)
    var_20 = list(var_19)
    var_21 = 'self'
    var_22 = 'MyClass'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = module_1.arg()
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.arg()
    var_29 = 'None'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_1.arg()
    var_33 = [var_25, var_28, var_32]
    var_34 = True
    var_35 = var_0.func_ann(var_1, var_33, has_self=var_34, cls_method=var_18)
    var_36 = list(var_35)
    var_37 = var_0.func_ann(var_1, var_33, has_self=var_34, cls_method=var_34)
    var_38 = list(var_37)
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.arg()
    var_42 = '*'
    var_43 = None
    var_44 = module_1.arg()
    var_45 = module_1.Load()
    var_46 = module_1.Name()
    var_47 = module_1.arg()
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = module_1.arg()
    var_51 = [var_41, var_44, var_47, var_50]
    var_52 = var_0.func_ann(var_1, var_51, has_self=var_18, cls_method=var_18)
    var_53 = list(var_52)
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = module_1.arg()
    var_57 = module_1.arg()
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = module_1.arg()
    var_61 = [var_56, var_57, var_60]
    var_62 = var_0.func_ann(var_1, var_61, has_self=var_18, cls_method=var_18)
    var_63 = list(var_62)
    var_64 = 'test_module.MyClass'
    var_65 = module_1.Load()
    var_66 = module_1.Name()
    var_67 = module_1.arg()
    var_68 = module_1.Load()
    var_69 = module_1.Name()
    var_70 = module_1.arg()
    var_71 = module_1.Load()
    var_72 = module_1.Name()
    var_73 = module_1.arg()
    var_74 = [var_67, var_70, var_73]
    var_75 = var_0.func_ann(var_1, var_74, has_self=var_34, cls_method=var_18)
    var_76 = list(var_75)
    var_77 = 'items'
    var_78 = 'List'
    var_79 = module_1.Load()
    var_80 = module_1.Name()
    var_81 = module_1.Load()
    var_82 = module_1.Name()
    var_83 = module_1.Load()
    var_84 = module_1.Subscript()
    var_85 = module_1.arg()
    var_86 = 'Dict'
    var_87 = module_1.Load()
    var_88 = module_1.Name()
    var_89 = module_1.Load()
    var_90 = module_1.Name()
    var_91 = module_1.Load()
    var_92 = module_1.Name()
    var_93 = [var_90, var_92]
    var_94 = module_1.Load()
    var_95 = module_1.Tuple()
    var_96 = module_1.Load()
    var_97 = module_1.Subscript()
    var_98 = module_1.arg()
    var_99 = [var_85, var_98]
    var_100 = var_0.func_ann(var_1, var_99, has_self=var_18, cls_method=var_18)
    var_101 = list(var_100)
    var_102 = []
    var_103 = var_0.func_ann(var_1, var_102, has_self=var_18, cls_method=var_18)
    var_104 = list(var_103)



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.func1'
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_3)
    var_9 = None
    var_10 = 'test_module'
    var_11 = False
    var_12 = var_0.func_api(var_10, var_1, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_13 = 'test_module.func2'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = 'z'
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = var_0.func_api(var_10, var_13, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_21 = 'test_module.func3'
    var_22 = []
    var_23 = 'a'
    var_24 = 'b'
    var_25 = []
    var_26 = []
    var_27 = 1
    var_28 = 'test'
    var_29 = var_0.func_api(var_10, var_21, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_30 = 'test_module.func4'
    var_31 = []
    var_32 = []
    var_33 = 'args'
    var_34 = []
    var_35 = []
    var_36 = 'kwargs'
    var_37 = []
    var_38 = var_0.func_api(var_10, var_30, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_39 = 'test_module.func5'
    var_40 = []
    var_41 = []
    var_42 = 2
    var_43 = []
    var_44 = var_0.func_api(var_10, var_39, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_45 = 'test_module.func6'
    var_46 = []
    var_47 = []
    var_48 = []
    var_49 = var_0.func_api(var_10, var_45, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_50 = 'test_module.Class.method'
    var_51 = []
    var_52 = 'self'
    var_53 = []
    var_54 = []
    var_55 = []
    var_56 = True
    var_57 = var_0.func_api(var_10, var_50, var_8, var_9, has_self=var_56, cls_method=var_11)
    var_58 = 'test_module.Class.class_method'
    var_59 = []
    var_60 = 'cls'
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = True
    var_65 = True
    var_66 = var_0.func_api(var_10, var_58, var_8, var_9, has_self=var_64, cls_method=var_65)
    var_67 = 'test_module.func7'
    var_68 = []
    var_69 = []
    var_70 = []
    var_71 = []
    var_72 = 'int'
    var_73 = module_1.Load()
    var_74 = var_0.func_api(var_10, var_67, var_8, var_9, has_self=var_11, cls_method=var_11)
    var_75 = 'test_module.func8'
    var_76 = 'c'
    var_77 = 'd'
    var_78 = 'e'
    var_79 = 10
    var_80 = 20
    var_81 = var_0.func_api(var_10, var_75, var_8, var_9, has_self=var_11, cls_method=var_11)



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\ndef simple_function():\n    """Simple function docstring."""\n    pass\n\nasync def async_function():\n    """Async function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        pass\n    \n    @classmethod\n    def class_method(cls):\n        pass\n    \n    @staticmethod\n    def static_method():\n        pass\n    \n    class InnerClass:\n        """Inner class docstring."""\n        pass\n'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = module_0.Parser(var_4, var_5, var_6)
    var_8 = 'another_module'
    var_9 = '\nclass BaseClass:\n    pass\n\nclass DerivedClass(BaseClass):\n    """Derived class with base."""\n    pass\n'
    var_10 = var_7.parse(var_8, var_9)
    var_11 = module_0.Parser()
    var_12 = 'module_with_decorators'
    var_13 = '\n@property\ndef computed_property():\n    """Property docstring."""\n    return 42\n\n@classmethod\n@staticmethod\ndef multi_decorated():\n    pass\n'
    var_14 = var_11.parse(var_12, var_13)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'always'
    var_4 = 'List'
    var_5 = module_1.Load()
    var_6 = 1
    var_7 = module_1.Load()
    var_8 = 0
    var_9 = 'Union'
    var_10 = module_1.Load()
    var_11 = 'int'
    var_12 = module_1.Load()
    var_13 = 'str'
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'Optional'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'Custom'
    var_22 = module_1.Load()
    var_23 = 'T'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = 'module'
    var_27 = module_1.Load()
    var_28 = 'Class'
    var_29 = module_1.Load()
    var_30 = module_1.Load()
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()
    var_35 = module_1.Load()
    var_36 = module_1.Load()
    var_37 = module_1.Load()
    var_38 = 'bool'
    var_39 = module_1.Load()
    var_40 = module_1.Load()
    var_41 = module_1.Load()



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = '\nclass MockClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    \n    @classmethod\n    def class_method(cls):\n        """Class method docstring."""\n        pass\n'
    var_3 = var_1.parse(var_0, var_2)
    var_4 = 'other_module'
    var_5 = '\ndef some_function():\n    pass\n'
    var_6 = var_1.parse(var_4, var_5)
    var_7 = 'nested.module'
    var_8 = '\nclass Outer:\n    class Inner:\n        """Inner class doc."""\n        pass\n'
    var_9 = var_1.parse(var_7, var_8)
    var_10 = 'empty_module'
    var_11 = '\nclass NoDoc:\n    pass\n'
    var_12 = var_1.parse(var_10, var_11)
    var_13 = ''
    var_14 = 'doctest_module'
    var_15 = '\ndef func_with_doctest():\n    """This is a doctest.\n    \n    >>> func_with_doctest()\n    \'result\'\n    """\n    return \'result\'\n'
    var_16 = var_1.parse(var_14, var_15)



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = var_6.ctx
    var_8 = 'test_module.name1'
    var_9 = 'typing.List[int]'
    var_10 = {var_8: var_9}
    var_11 = ''
    var_12 = module_0.Resolver(var_0, var_10, var_11)
    var_13 = 'name1'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = var_12.visit_Name(var_15)
    var_17 = var_16.value
    var_18 = var_16.slice
    var_19 = 'test_module.T'
    var_20 = "typing.TypeVar('T')"
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_0, var_21, var_11)
    var_23 = 'T'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = var_22.visit_Name(var_25)
    var_27 = {}
    var_28 = module_0.Resolver(var_0, var_27, var_11)
    var_29 = 'some_name'
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = var_28.visit_Name(var_31)
    var_33 = 'test_module.name2'
    var_34 = {var_33: var_33}
    var_35 = module_0.Resolver(var_0, var_34, var_11)
    var_36 = 'name2'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = var_35.visit_Name(var_38)
    var_40 = 'test_module.outer'
    var_41 = 'test_module.inner'
    var_42 = 'typing.Dict[str, int]'
    var_43 = {var_40: var_42, var_41: var_40}
    var_44 = module_0.Resolver(var_0, var_43, var_11)
    var_45 = 'inner'
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = var_44.visit_Name(var_47)
    var_49 = var_48.value
    var_50 = var_48.slice
    var_51 = var_48.slice.elts
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = 0
    var_54 = var_48.slice.elts[var_53]
    var_55 = 1
    var_56 = var_48.slice.elts[var_55]



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a regular docstring.'
    var_1 = module_0.doctest(var_0)
    var_2 = ">>> print('hello')"
    var_3 = module_0.doctest(var_2)
    assert var_3 == "```python\n>>> print('hello')\n```"
    var_4 = '>>> x = 1\n>>> print(x)'
    var_5 = module_0.doctest(var_4)
    assert var_5 == '```python\n>>> x = 1\n>>> print(x)\n```'
    var_6 = 'Some text.\n>>> code()\nMore text.'
    var_7 = module_0.doctest(var_6)
    assert var_7 == 'Some text.\n```python\n>>> code()\n```\nMore text.'
    var_8 = '>>> for i in range(3):\n...     print(i)'
    var_9 = module_0.doctest(var_8)
    assert var_9 == '```python\n>>> for i in range(3):\n...     print(i)\n```'
    var_10 = '>>> first()\nText\n>>> second()'
    var_11 = module_0.doctest(var_10)
    assert var_11 == '```python\n>>> first()\n```\nText\n```python\n>>> second()\n```'
    var_12 = ''
    var_13 = module_0.doctest(var_12)
    assert var_13 == ''
    var_14 = '   \n\t\n'
    var_15 = module_0.doctest(var_14)
    assert var_15 == '   \n\t\n'



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = {}
    var_7 = module_0.Resolver(var_0, var_6)
    var_8 = 'not a valid python expression!'
    var_9 = module_1.Constant()
    var_10 = var_7.visit_Constant(var_9)
    var_11 = {}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'test_name'
    var_14 = module_1.Constant()
    var_15 = var_12.visit_Constant(var_14)
    var_16 = 'test_module.test_name'
    var_17 = 'other_name'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = module_1.Constant()
    var_21 = var_19.visit_Constant(var_20)
    var_22 = 'test_module.TypeVar'
    var_23 = 'typing.TypeVar'
    var_24 = {var_22: var_23}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = 'TypeVar'
    var_27 = module_1.Constant()
    var_28 = var_25.visit_Constant(var_27)
    var_29 = {}
    var_30 = module_0.Resolver(var_0, var_29)
    var_31 = ''
    var_32 = module_1.Constant()
    var_33 = var_30.visit_Constant(var_32)
    var_34 = {}
    var_35 = module_0.Resolver(var_0, var_34)
    var_36 = 'a + b'
    var_37 = module_1.Constant()
    var_38 = var_35.visit_Constant(var_37)
    var_39 = var_38.left
    var_40 = var_38.right



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TYPE_ALIAS'
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'list'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1
    var_10 = 'CONSTANT'
    var_11 = 42
    var_12 = module_1.Constant()
    var_13 = 'int'
    var_14 = 'ANOTHER'
    var_15 = 'string'
    var_16 = module_1.Constant()
    var_17 = '__all__'
    var_18 = 'func1'
    var_19 = module_1.Constant()
    var_20 = 'Class1'
    var_21 = module_1.Constant()
    var_22 = [var_19, var_21]
    var_23 = module_1.Load()
    var_24 = module_1.Tuple()
    var_25 = 'func2'
    var_26 = module_1.Constant()
    var_27 = 'attr'
    var_28 = module_1.Constant()
    var_29 = [var_26, var_28]
    var_30 = module_1.Load()
    var_31 = module_1.List()
    var_32 = 'variable'
    var_33 = 3.14
    var_34 = module_1.Constant()
    var_35 = 'NO_VALUE'
    var_36 = 'str'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = None
    var_40 = 'a'
    var_41 = 'b'
    var_42 = module_1.Constant()
    var_43 = 'obj'
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = 2
    var_47 = module_1.Constant()



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'another_var'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = 'test_module.another_var'
    var_14 = 'no_type'
    var_15 = 3.14
    var_16 = module_1.Constant()
    var_17 = 'test_module.no_type'
    var_18 = '__all__'
    var_19 = 'func1'
    var_20 = module_1.Constant()
    var_21 = 'ClassA'
    var_22 = module_1.Constant()
    var_23 = [var_20, var_22]
    var_24 = module_1.Load()
    var_25 = module_1.Tuple()
    var_26 = 'var1'
    var_27 = module_1.Constant()
    var_28 = 'CONST2'
    var_29 = module_1.Constant()
    var_30 = [var_27, var_29]
    var_31 = module_1.Load()
    var_32 = module_1.List()
    var_33 = 'COMPLEX_CONST'
    var_34 = 'list'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = module_1.Constant()
    var_38 = 2
    var_39 = module_1.Constant()
    var_40 = [var_37, var_39]
    var_41 = module_1.Load()
    var_42 = module_1.List()
    var_43 = 'obj'
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = 'attr'
    var_47 = 5
    var_48 = module_1.Constant()
    var_49 = 'a'
    var_50 = 'b'
    var_51 = 10
    var_52 = module_1.Constant()
    var_53 = 'no_value'
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = None
    var_57 = 'OVERWRITE'
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = module_1.Constant()
    var_61 = module_1.Load()
    var_62 = module_1.Name()
    var_63 = 'new'
    var_64 = module_1.Constant()



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 0
    var_4 = 'Union[int, str]'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = 'Optional[int]'
    var_9 = module_1.parse(var_8)
    var_10 = var_9.body[var_3]
    var_11 = var_10.value
    var_12 = 'List[int]'
    var_13 = module_1.parse(var_12)
    var_14 = var_13.body[var_3]
    var_15 = var_14.value
    var_16 = 'Custom[int]'
    var_17 = module_1.parse(var_16)
    var_18 = var_17.body[var_3]
    var_19 = var_18.value
    var_20 = 'Union[int]'
    var_21 = module_1.parse(var_20)
    var_22 = var_21.body[var_3]
    var_23 = var_22.value
    var_24 = 'Union[int, str, float]'
    var_25 = module_1.parse(var_24)
    var_26 = var_25.body[var_3]
    var_27 = var_26.value



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'test_module'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)
    var_11 = 'self'
    var_12 = None
    var_13 = module_1.arg()
    var_14 = 'str'
    var_15 = module_1.Load()
    var_16 = module_1.Name()
    var_17 = module_1.arg()
    var_18 = [var_13, var_17]
    var_19 = True
    var_20 = var_0.func_ann(var_7, var_18, has_self=var_19, cls_method=var_8)
    var_21 = list(var_20)
    var_22 = 'cls'
    var_23 = 'type'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = module_1.arg()
    var_27 = 'float'
    var_28 = module_1.Load()
    var_29 = module_1.Name()
    var_30 = module_1.arg()
    var_31 = [var_26, var_30]
    var_32 = var_0.func_ann(var_7, var_31, has_self=var_19, cls_method=var_19)
    var_33 = list(var_32)
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = module_1.arg()
    var_37 = '*'
    var_38 = module_1.arg()
    var_39 = 'y'
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = module_1.arg()
    var_43 = [var_36, var_38, var_42]
    var_44 = var_0.func_ann(var_7, var_43, has_self=var_8, cls_method=var_8)
    var_45 = list(var_44)
    var_46 = module_1.arg()
    var_47 = [var_46]
    var_48 = var_0.func_ann(var_7, var_47, has_self=var_8, cls_method=var_8)
    var_49 = list(var_48)
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = module_1.arg()
    var_53 = 'return'
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = module_1.arg()
    var_57 = [var_52, var_56]
    var_58 = var_0.func_ann(var_7, var_57, has_self=var_8, cls_method=var_8)
    var_59 = list(var_58)
    var_60 = 'test_module.int'
    var_61 = 'builtins.int'
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = module_1.arg()
    var_65 = [var_64]
    var_66 = var_0.func_ann(var_7, var_65, has_self=var_8, cls_method=var_8)
    var_67 = list(var_66)
    var_68 = 'MyClass'
    var_69 = module_1.Load()
    var_70 = module_1.Name()
    var_71 = module_1.arg()
    var_72 = module_1.Load()
    var_73 = module_1.Name()
    var_74 = module_1.arg()
    var_75 = [var_71, var_74]
    var_76 = var_0.func_ann(var_7, var_75, has_self=var_19, cls_method=var_8)
    var_77 = list(var_76)
    var_78 = 'a'
    var_79 = module_1.Load()
    var_80 = module_1.Name()
    var_81 = module_1.arg()
    var_82 = 'b'
    var_83 = module_1.Load()
    var_84 = module_1.Name()
    var_85 = module_1.arg()
    var_86 = 'c'
    var_87 = module_1.Load()
    var_88 = module_1.Name()
    var_89 = module_1.arg()
    var_90 = [var_81, var_85, var_89]
    var_91 = var_0.func_ann(var_7, var_90, has_self=var_8, cls_method=var_8)
    var_92 = list(var_91)



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'os'
    var_5 = 'operating_system'
    var_6 = 'sys'
    var_7 = None
    var_8 = 'submodule'
    var_9 = 'func'
    var_10 = 1
    var_11 = 'parent.child'
    var_12 = 'package.module'
    var_13 = 'Class'
    var_14 = 'Cls'
    var_15 = 0
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'OrderedDict'
    var_19 = 'ODict'
    var_20 = 'new_module'
    var_21 = 'existing'



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not a valid name syntax'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'test_name'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = var_11.ctx
    var_13 = 'test_module.test_name'
    var_14 = 'mapped_name'
    var_15 = {var_13: var_14}
    var_16 = module_0.Resolver(var_0, var_15)
    var_17 = module_1.Constant()
    var_18 = var_16.visit_Constant(var_17)
    var_19 = {}
    var_20 = 'SelfType'
    var_21 = module_0.Resolver(var_0, var_19, var_20)
    var_22 = module_1.Constant()
    var_23 = var_21.visit_Constant(var_22)
    var_24 = 'Union[int, str]'
    var_25 = module_1.Constant()
    var_26 = var_21.visit_Constant(var_25)
    var_27 = var_26.left
    var_28 = var_26.op
    var_29 = var_26.right



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = {}
    var_7 = module_0.Resolver(var_0, var_6)
    var_8 = 'not a valid python expression!'
    var_9 = module_1.Constant()
    var_10 = var_7.visit_Constant(var_9)
    var_11 = {}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'some_name'
    var_14 = module_1.Constant()
    var_15 = var_12.visit_Constant(var_14)
    var_16 = {}
    var_17 = module_0.Resolver(var_0, var_16)
    var_18 = 'Union[int, str]'
    var_19 = module_1.Constant()
    var_20 = var_17.visit_Constant(var_19)
    var_21 = var_20.left
    var_22 = var_20.right
    var_23 = 'test_module.some_name'
    var_24 = 'mapped_name'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = module_1.Constant()
    var_28 = var_26.visit_Constant(var_27)
    var_29 = {}
    var_30 = 'SelfType'
    var_31 = module_0.Resolver(var_0, var_29, var_30)
    var_32 = module_1.Constant()
    var_33 = var_31.visit_Constant(var_32)
    var_34 = 'test_module.T'
    var_35 = "typing.TypeVar('T')"
    var_36 = {var_34: var_35}
    var_37 = module_0.Resolver(var_0, var_36)
    var_38 = 'T'
    var_39 = module_1.Constant()
    var_40 = var_37.visit_Constant(var_39)



# Parsed testcases at query #22
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'
    var_3 = 3.14
    var_4 = module_0.Constant()
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'float'
    var_6 = 'hello'
    var_7 = module_0.Constant()
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'str'
    var_9 = True
    var_10 = module_0.Constant()
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'bool'
    var_12 = None
    var_13 = module_0.Constant()
    var_14 = module_1.const_type(var_13)
    assert var_14 == 'NoneType'
    var_15 = module_0.Constant()
    var_16 = 2
    var_17 = module_0.Constant()
    var_18 = [var_15, var_17]
    var_19 = module_0.Tuple()
    var_20 = module_1.const_type(var_19)
    assert var_20 == 'tuple[int, int]'
    var_21 = 'a'
    var_22 = module_0.Constant()
    var_23 = 'b'
    var_24 = module_0.Constant()
    var_25 = [var_22, var_24]
    var_26 = module_0.Tuple()
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'tuple[str, str]'
    var_28 = module_0.Constant()
    var_29 = module_0.Constant()
    var_30 = [var_28, var_29]
    var_31 = module_0.Tuple()
    var_32 = module_1.const_type(var_31)
    assert var_32 == 'tuple[Any, Any]'
    var_33 = []
    var_34 = module_0.Tuple()
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'tuple'
    var_36 = module_0.Constant()
    var_37 = module_0.Constant()
    var_38 = [var_36, var_37]
    var_39 = module_0.List()
    var_40 = module_1.const_type(var_39)
    assert var_40 == 'list[float, float]'
    var_41 = module_0.Constant()
    var_42 = False
    var_43 = module_0.Constant()
    var_44 = [var_41, var_43]
    var_45 = module_0.List()
    var_46 = module_1.const_type(var_45)
    assert var_46 == 'list[bool, bool]'
    var_47 = []
    var_48 = module_0.List()
    var_49 = module_1.const_type(var_48)
    assert var_49 == 'list'
    var_50 = module_0.Constant()
    var_51 = module_0.Constant()
    var_52 = [var_50, var_51]
    var_53 = module_0.Set()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'set[int, int]'
    var_55 = []
    var_56 = module_0.Set()
    var_57 = module_1.const_type(var_56)
    assert var_57 == 'set'
    var_58 = 'key'
    var_59 = module_0.Constant()
    var_60 = [var_59]
    var_61 = module_0.Constant()
    var_62 = [var_61]
    var_63 = module_0.Dict()
    var_64 = module_1.const_type(var_63)
    assert var_64 == 'dict[str, int]'
    var_65 = module_0.Constant()
    var_66 = [var_65]
    var_67 = 'value'
    var_68 = module_0.Constant()
    var_69 = [var_68]
    var_70 = module_0.Dict()
    var_71 = module_1.const_type(var_70)
    assert var_71 == 'dict[int, str]'
    var_72 = []
    var_73 = []
    var_74 = module_0.Dict()
    var_75 = module_1.const_type(var_74)
    assert var_75 == 'dict'
    var_76 = 'int'
    var_77 = module_0.Load()
    var_78 = module_0.Name()
    var_79 = []
    var_80 = []
    var_81 = module_0.Call(*var_79)
    var_82 = module_1.const_type(var_81)
    assert var_82 == 'int'
    var_83 = 'str'
    var_84 = module_0.Load()
    var_85 = module_0.Name()
    var_86 = []
    var_87 = []
    var_88 = module_0.Call(*var_86)
    var_89 = module_1.const_type(var_88)
    assert var_89 == 'str'
    var_90 = 'bool'
    var_91 = module_0.Load()
    var_92 = module_0.Name()
    var_93 = []
    var_94 = []
    var_95 = module_0.Call(*var_93)
    var_96 = module_1.const_type(var_95)
    assert var_96 == 'bool'
    var_97 = 'float'
    var_98 = module_0.Load()
    var_99 = module_0.Name()
    var_100 = []
    var_101 = []
    var_102 = module_0.Call(*var_100)
    var_103 = module_1.const_type(var_102)
    assert var_103 == 'float'
    var_104 = 'complex'
    var_105 = module_0.Load()
    var_106 = module_0.Name()
    var_107 = []
    var_108 = []
    var_109 = module_0.Call(*var_107)
    var_110 = module_1.const_type(var_109)
    assert var_110 == 'complex'
    var_111 = 'list'
    var_112 = module_0.Load()
    var_113 = module_0.Name()
    var_114 = []
    var_115 = []
    var_116 = module_0.Call(*var_114)
    var_117 = module_1.const_type(var_116)
    assert var_117 == 'list'
    var_118 = 'dict'
    var_119 = module_0.Load()
    var_120 = module_0.Name()
    var_121 = []
    var_122 = []
    var_123 = module_0.Call(*var_121)
    var_124 = module_1.const_type(var_123)
    assert var_124 == 'dict'
    var_125 = 'tuple'
    var_126 = module_0.Load()
    var_127 = module_0.Name()
    var_128 = []
    var_129 = []
    var_130 = module_0.Call(*var_128)
    var_131 = module_1.const_type(var_130)
    assert var_131 == 'tuple'
    var_132 = 'typing'
    var_133 = module_0.Load()
    var_134 = module_0.Name()
    var_135 = 'List'
    var_136 = module_0.Load()
    var_137 = module_0.Attribute()
    var_138 = []
    var_139 = []
    var_140 = module_0.Call(*var_138)
    var_141 = module_1.const_type(var_140)
    assert var_141 == 'List'
    var_142 = 'x'
    var_143 = module_0.Load()
    var_144 = module_0.Name()
    var_145 = module_1.const_type(var_144)
    assert var_145 == 'Any'
    var_146 = module_0.Constant()
    var_147 = module_0.BitOr()
    var_148 = module_0.Constant()
    var_149 = module_0.BinOp()
    var_150 = module_1.const_type(var_149)
    assert var_150 == 'Any'
    var_151 = module_0.Load()
    var_152 = module_0.Name()
    var_153 = module_0.Constant()
    var_154 = module_0.Load()
    var_155 = module_0.Subscript()
    var_156 = module_1.const_type(var_155)
    assert var_156 == 'Any'



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not valid python code'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'List[int]'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = 'SomeType'
    var_13 = module_1.Constant()
    var_14 = var_2.visit_Constant(var_13)
    var_15 = 'test_module.SomeType'
    var_16 = 'typing.List[int]'
    var_17 = var_2.visit_Constant(var_13)
    var_18 = var_17.value
    var_19 = module_1.Constant()
    var_20 = var_2.visit_Constant(var_19)



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.func1'
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = module_1.arguments(*var_6)
    var_12 = None
    var_13 = False
    var_14 = var_0.func_api(var_1, var_4, var_11, var_12, has_self=var_13, cls_method=var_13)
    var_15 = 'test_module.func2'
    var_16 = []
    var_17 = 'x'
    var_18 = module_1.arg()
    var_19 = 'y'
    var_20 = module_1.arg()
    var_21 = [var_18, var_20]
    var_22 = []
    var_23 = []
    var_24 = []
    var_25 = module_1.arguments(*var_21)
    var_26 = var_0.func_api(var_1, var_15, var_25, var_12, has_self=var_13, cls_method=var_13)
    var_27 = 'test_module.Class.method'
    var_28 = []
    var_29 = 'self'
    var_30 = module_1.arg()
    var_31 = module_1.arg()
    var_32 = [var_30, var_31]
    var_33 = []
    var_34 = []
    var_35 = []
    var_36 = module_1.arguments(*var_32)
    var_37 = True
    var_38 = var_0.func_api(var_1, var_27, var_36, var_12, has_self=var_37, cls_method=var_13)
    var_39 = 'test_module.Class.class_method'
    var_40 = []
    var_41 = 'cls'
    var_42 = module_1.arg()
    var_43 = module_1.arg()
    var_44 = [var_42, var_43]
    var_45 = []
    var_46 = []
    var_47 = []
    var_48 = module_1.arguments(*var_44)
    var_49 = var_0.func_api(var_1, var_39, var_48, var_12, has_self=var_37, cls_method=var_37)
    var_50 = 'test_module.func3'
    var_51 = []
    var_52 = module_1.arg()
    var_53 = module_1.arg()
    var_54 = [var_52, var_53]
    var_55 = []
    var_56 = []
    var_57 = module_1.Constant()
    var_58 = 2
    var_59 = module_1.Constant()
    var_60 = [var_57, var_59]
    var_61 = module_1.arguments(*var_54)
    var_62 = var_0.func_api(var_1, var_50, var_61, var_12, has_self=var_13, cls_method=var_13)
    var_63 = 'test_module.func4'
    var_64 = []
    var_65 = module_1.arg()
    var_66 = [var_65]
    var_67 = 'args'
    var_68 = module_1.arg()
    var_69 = 'kw1'
    var_70 = module_1.arg()
    var_71 = [var_70]
    var_72 = 'default'
    var_73 = module_1.Constant()
    var_74 = [var_73]
    var_75 = 'kwargs'
    var_76 = module_1.arg()
    var_77 = []
    var_78 = module_1.arguments(*var_66)
    var_79 = var_0.func_api(var_1, var_63, var_78, var_12, has_self=var_13, cls_method=var_13)
    var_80 = 'test_module.func5'
    var_81 = []
    var_82 = module_1.arg()
    var_83 = [var_82]
    var_84 = []
    var_85 = []
    var_86 = []
    var_87 = module_1.arguments(*var_83)
    var_88 = 'int'
    var_89 = module_1.Load()
    var_90 = module_1.Name()
    var_91 = var_0.func_api(var_1, var_80, var_87, var_90, has_self=var_13, cls_method=var_13)
    var_92 = 'test_module.func6'
    var_93 = module_1.arg()
    var_94 = [var_93]
    var_95 = module_1.arg()
    var_96 = [var_95]
    var_97 = []
    var_98 = []
    var_99 = []
    var_100 = module_1.arguments(*var_96)
    var_101 = var_0.func_api(var_1, var_92, var_100, var_90, has_self=var_13, cls_method=var_13)
    var_102 = 'test_module.func7'
    var_103 = []
    var_104 = module_1.arg()
    var_105 = [var_104]
    var_106 = module_1.arg()
    var_107 = 'kw2'
    var_108 = module_1.arg()
    var_109 = [var_106, var_108]
    var_110 = module_1.Constant()
    var_111 = module_1.Constant()
    var_112 = [var_110, var_111]
    var_113 = []
    var_114 = module_1.arguments(*var_105)
    var_115 = var_0.func_api(var_1, var_102, var_114, var_90, has_self=var_13, cls_method=var_13)



# Parsed testcases at query #25
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not a valid expression'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'some_name'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = 'test_module.some_name'
    var_13 = 'other_name'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = module_1.Constant()
    var_17 = var_15.visit_Constant(var_16)
    var_18 = 'some_name[0]'
    var_19 = module_1.Constant()
    var_20 = var_15.visit_Constant(var_19)
    var_21 = var_20.value



# Parsed testcases at query #26
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 'not a valid name syntax'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'SomeType'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = 'test_module.SomeType'
    var_13 = 'typing.List[int]'
    var_14 = var_2.visit_Constant(var_10)
    var_15 = var_14.value
    var_16 = var_2.visit_Constant(var_10)
    var_17 = 'Union[int, str]'
    var_18 = module_1.Constant()
    var_19 = var_2.visit_Constant(var_18)
    var_20 = var_19.left
    var_21 = var_19.op
    var_22 = var_19.right



