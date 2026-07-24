####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_16 = module_1.FunctionDef(*var_13)
    var_17 = var_0.api(var_1, var_16)
    var_18 = module_0.Parser()
    var_19 = 'test_async_func'
    var_20 = []
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = []
    var_25 = module_1.arguments(*var_21)
    var_26 = []
    var_27 = []
    var_28 = module_1.AsyncFunctionDef(*var_25)
    var_29 = var_18.api(var_1, var_28)
    var_30 = module_0.Parser()
    var_31 = 'TestClass'
    var_32 = []
    var_33 = []
    var_34 = []
    var_35 = module_1.ClassDef()
    var_36 = var_30.api(var_1, var_35)
    var_37 = module_0.Parser()
    var_38 = 'decorated_func'
    var_39 = []
    var_40 = []
    var_41 = []
    var_42 = []
    var_43 = []
    var_44 = module_1.arguments(*var_40)
    var_45 = []
    var_46 = 'decorator'
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = [var_48]
    var_50 = module_1.FunctionDef(*var_44)
    var_51 = var_37.api(var_1, var_50)
    var_52 = module_0.Parser()
    var_53 = 'ChildClass'
    var_54 = 'ParentClass'
    var_55 = module_1.Load()
    var_56 = module_1.Name()
    var_57 = [var_56]
    var_58 = []
    var_59 = []
    var_60 = module_1.ClassDef()
    var_61 = var_52.api(var_1, var_60)
    var_62 = module_0.Parser()
    var_63 = 'OuterClass'
    var_64 = []
    var_65 = 'InnerClass'
    var_66 = []
    var_67 = []
    var_68 = []
    var_69 = module_1.ClassDef()
    var_70 = [var_69]
    var_71 = []
    var_72 = module_1.ClassDef()
    var_73 = var_62.api(var_1, var_72)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'module.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True
    var_7 = module_0.Parser()
    var_8 = set()
    var_9 = 'module._private_name'
    var_10 = var_7.is_public(var_9)
    assert var_10 is False
    var_11 = module_0.Parser()
    var_12 = set()
    var_13 = var_11.is_public(var_4)
    assert var_13 is True
    var_14 = module_0.Parser()
    var_15 = 'submodule'
    var_16 = {var_15}
    var_17 = 'module.submodule.item'
    var_18 = var_14.is_public(var_17)
    assert var_18 is True
    var_19 = module_0.Parser()
    var_20 = set()
    var_21 = 'module.submodule'
    var_22 = 'module.submodule.public_item'
    var_23 = var_19.is_public(var_21)
    assert var_23 is True
    var_24 = module_0.Parser()
    var_25 = set()
    var_26 = 'module._private'
    var_27 = 'module._private.public_item'
    var_28 = var_24.is_public(var_26)
    assert var_28 is False



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = 'other'
    var_10 = module_1.Load()
    var_11 = 'attr'
    var_12 = module_1.Load()
    var_13 = {}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = 1
    var_16 = module_1.Load()



# Parsed testcases at query #4
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
    var_10 = 2
    var_11 = module_1.Constant()
    var_12 = module_1.Expr()
    var_13 = [var_9, var_12]
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = 'y'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = [var_20]
    var_22 = 3
    var_23 = module_1.Constant()
    var_24 = module_1.Assign()
    var_25 = [var_24]
    var_26 = 4
    var_27 = module_1.Constant()
    var_28 = module_1.Expr()
    var_29 = [var_28]
    var_30 = module_1.If()
    var_31 = [var_30]
    var_32 = module_0.walk_body(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 0
    var_36 = var_33[var_35]
    var_37 = var_33[var_16]
    var_38 = 'z'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = [var_40]
    var_42 = 5
    var_43 = module_1.Constant()
    var_44 = module_1.Assign()
    var_45 = [var_44]
    var_46 = []
    var_47 = 6
    var_48 = module_1.Constant()
    var_49 = module_1.Expr()
    var_50 = [var_49]
    var_51 = 7
    var_52 = module_1.Constant()
    var_53 = module_1.Expr()
    var_54 = [var_53]
    var_55 = module_1.Try()
    var_56 = [var_55]
    var_57 = module_0.walk_body(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 3
    var_60 = True
    var_61 = module_1.Constant()
    var_62 = 'a'
    var_63 = module_1.Load()
    var_64 = module_1.Name()
    var_65 = [var_64]
    var_66 = 8
    var_67 = module_1.Constant()
    var_68 = module_1.Assign()
    var_69 = [var_68]
    var_70 = []
    var_71 = []
    var_72 = []
    var_73 = module_1.Try()
    var_74 = [var_73]
    var_75 = []
    var_76 = module_1.If()
    var_77 = [var_76]
    var_78 = module_0.walk_body(var_77)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 1
    var_81 = var_79[var_35]
    var_82 = 'b'
    var_83 = module_1.Load()
    var_84 = module_1.Name()
    var_85 = [var_84]
    var_86 = 9
    var_87 = module_1.Constant()
    var_88 = module_1.Assign()
    var_89 = True
    var_90 = module_1.Constant()
    var_91 = 10
    var_92 = module_1.Constant()
    var_93 = module_1.Expr()
    var_94 = [var_93]
    var_95 = []
    var_96 = module_1.If()
    var_97 = 'c'
    var_98 = module_1.Load()
    var_99 = module_1.Name()
    var_100 = [var_99]
    var_101 = 11
    var_102 = module_1.Constant()
    var_103 = module_1.Assign()
    var_104 = [var_103]
    var_105 = []
    var_106 = []
    var_107 = []
    var_108 = module_1.Try()
    var_109 = [var_88, var_96, var_108]
    var_110 = module_0.walk_body(var_109)
    var_111 = list(var_110)
    var_112 = len(var_111)
    assert var_112 == 3
    var_113 = var_111[var_35]
    var_114 = var_111[var_89]
    var_115 = var_111[var_10]



# Parsed testcases at query #5
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
    var_17 = []
    var_18 = 'arg1'
    var_19 = 'int'
    var_20 = module_0.Load()
    var_21 = module_0.Name()
    var_22 = module_0.arg()
    var_23 = 'arg2'
    var_24 = 'str'
    var_25 = module_0.Load()
    var_26 = module_0.Name()
    var_27 = module_0.arg()
    var_28 = [var_22, var_27]
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
    var_43 = module_0.Load()
    var_44 = module_0.Name()
    var_45 = module_0.arg()
    var_46 = [var_45]
    var_47 = []
    var_48 = []
    var_49 = 10
    var_50 = module_0.Constant()
    var_51 = [var_50]
    var_52 = module_0.arguments(*var_46)
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
    var_78 = module_0.Load()
    var_79 = module_0.Name()
    var_80 = module_0.arg()
    var_81 = [var_77, var_80]
    var_82 = []
    var_83 = []
    var_84 = []
    var_85 = module_0.arguments(*var_81)
    var_86 = []
    var_87 = module_0.FunctionDef(*var_85)
    var_88 = 'test_module.method'
    var_89 = var_87.args
    var_90 = var_87.returns
    var_91 = True
    var_92 = 'classmethod_func'
    var_93 = []
    var_94 = 'cls'
    var_95 = module_0.arg()
    var_96 = module_0.Load()
    var_97 = module_0.Name()
    var_98 = module_0.arg()
    var_99 = [var_95, var_98]
    var_100 = []
    var_101 = []
    var_102 = []
    var_103 = module_0.arguments(*var_99)
    var_104 = 'classmethod'
    var_105 = module_0.Load()
    var_106 = module_0.Name()
    var_107 = [var_106]
    var_108 = module_0.FunctionDef(*var_103)
    var_109 = 'test_module.classmethod_func'
    var_110 = var_108.args
    var_111 = var_108.returns
    var_112 = True
    var_113 = True



# Parsed testcases at query #6
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
    var_6 = 1
    var_7 = None
    var_8 = 'attr2'
    var_9 = 'hello'
    var_10 = 'attr3'
    var_11 = 3.14



# Parsed testcases at query #7
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
    var_13 = 'test.typing.Optional'
    var_14 = 'typing.Optional'
    var_15 = {var_13: var_14}
    var_16 = module_0.Resolver(var_0, var_15)
    var_17 = 'Optional'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'test.typing.List'
    var_22 = 'typing.List'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = 'List'
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = 'test.typing.Dict'
    var_30 = 'typing.Dict'
    var_31 = {var_29: var_30}
    var_32 = module_0.Resolver(var_0, var_31)
    var_33 = 'Dict'
    var_34 = module_1.Load()
    var_35 = module_1.Load()
    var_36 = module_1.Load()
    var_37 = module_1.Load()
    var_38 = module_1.Load()



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = 'return'
    var_6 = 'str'
    var_7 = module_1.Load()
    var_8 = False
    var_9 = 'self'
    var_10 = 'TestClass'
    var_11 = module_1.Load()
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = True
    var_15 = 'cls'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = '*'
    var_21 = None
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = '**kwargs'
    var_25 = 'dict'
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = 'collections'
    var_6 = 'defaultdict'
    var_7 = 0
    var_8 = 'os.path'
    var_9 = 'join'
    var_10 = 1
    var_11 = 'typing'
    var_12 = 'List'
    var_13 = 'list'



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test_module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'test_module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'final'
    var_23 = {var_1: var_11, var_11: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'invalid name'
    var_5 = 'valid_name'
    var_6 = 'test.valid_name'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #12
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
    var_7 = 'test.valid_name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'valid_name'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #13
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
    var_7 = 'test'
    var_8 = False
    var_9 = module_0.Parser()
    var_10 = 'self'
    var_11 = 'TestClass'
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = True
    var_15 = module_0.Parser()
    var_16 = 'cls'
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = module_0.Parser()
    var_20 = module_1.Load()
    var_21 = '*'
    var_22 = None
    var_23 = module_1.Load()
    var_24 = module_0.Parser()
    var_25 = module_1.Load()
    var_26 = '**kwargs'
    var_27 = 'Any'
    var_28 = module_1.Load()
    var_29 = module_0.Parser()



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'typing.List'
    var_6 = module_1.Load()
    var_7 = 1
    var_8 = module_1.Load()
    var_9 = 'test.typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'typing.Union'
    var_14 = module_1.Load()
    var_15 = 2
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'test.typing.Optional'
    var_19 = 'Optional'
    var_20 = {var_18: var_19}
    var_21 = module_0.Resolver(var_0, var_20)
    var_22 = 'typing.Optional'
    var_23 = module_1.Load()
    var_24 = module_1.Load()
    var_25 = {}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = 'some.OtherType'
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST_A'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = 'CONST_B'
    var_9 = 3.14
    var_10 = 'float'
    var_11 = 'CONST_C'
    var_12 = 2
    var_13 = module_1.Load()
    var_14 = '__all__'
    var_15 = 'func1'
    var_16 = 'func2'
    var_17 = module_1.Load()
    var_18 = 'non_const'
    var_19 = 100
    var_20 = 'var1'
    var_21 = 'var2'
    var_22 = 200



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = module_0.Parser()
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'BaseClass'
    var_10 = module_1.Load()
    var_11 = []
    var_12 = var_6.class_api(var_7, var_8, var_3, var_11)
    var_13 = module_0.Parser()
    var_14 = 'test_module'
    var_15 = 'test_module.TestClass'
    var_16 = []
    var_17 = 'attr1'
    var_18 = 'int'
    var_19 = module_1.Load()
    var_20 = None
    var_21 = 1
    var_22 = 'attr2'
    var_23 = 'str'
    var_24 = module_1.Load()
    var_25 = var_13.class_api(var_14, var_15, var_16, var_11)
    var_26 = module_0.Parser()
    var_27 = 'test_module'
    var_28 = 'test_module.TestClass'
    var_29 = []
    var_30 = 'hello'
    var_31 = var_26.class_api(var_27, var_28, var_29, var_11)
    var_32 = module_0.Parser()
    var_33 = 'test_module'
    var_34 = 'test_module.TestEnum'
    var_35 = 'enum.Enum'
    var_36 = module_1.Load()
    var_37 = 'VALUE1'
    var_38 = 'VALUE2'
    var_39 = 2
    var_40 = var_32.class_api(var_33, var_34, var_29, var_11)
    var_41 = module_0.Parser()
    var_42 = 'test_module'
    var_43 = 'test_module.TestClass'
    var_44 = []
    var_45 = module_1.Load()
    var_46 = var_41.class_api(var_42, var_43, var_44, var_11)



# Parsed testcases at query #17
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
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = {}
    var_29 = module_0.Resolver(var_0, var_28)
    var_30 = 'CustomType'
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.compile()
    var_5 = var_0.compile()
    var_6 = module_0.Parser()
    var_7 = 'from typing import List\nx: List[int] = []'
    var_8 = var_6.parse(var_1, var_7)
    var_9 = var_6.compile()
    var_10 = var_6.compile()
    var_11 = module_0.Parser()
    var_12 = '\nclass MyClass:\n    def method(self): pass\n'
    var_13 = var_11.parse(var_1, var_12)
    var_14 = var_11.compile()
    var_15 = var_11.compile()
    var_16 = module_0.Parser()
    var_17 = 'CONSTANT = 42'
    var_18 = var_16.parse(var_1, var_17)
    var_19 = var_16.compile()
    var_20 = var_16.compile()
    var_21 = module_0.Parser()
    var_22 = '\ndef _private(): pass\nclass MyClass:\n    def __init__(self): pass\n'
    var_23 = var_21.parse(var_1, var_22)
    var_24 = var_21.compile()
    var_25 = var_21.compile()
    var_26 = module_0.Parser()
    var_27 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_28 = var_26.parse(var_1, var_27)
    var_29 = var_26.compile()
    var_30 = var_26.compile()
    var_31 = True
    var_32 = module_0.Parser(toc=var_31)
    var_33 = var_32.parse(var_1, var_2)
    var_34 = var_32.compile()
    var_35 = module_0.Parser()
    var_36 = var_35.parse(var_1, var_2)
    var_37 = var_35.compile()



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'self_ty'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = {var_1: var_1}
    var_17 = module_0.Resolver(var_0, var_16)
    var_18 = module_1.Load()
    var_19 = 'module.TypeVar'
    var_20 = 'typing.TypeVar'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = 'TypeVar'
    var_24 = module_1.Load()



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = 'test_module'
    var_6 = 'collections'
    var_7 = 'defaultdict'
    var_8 = 0
    var_9 = 'typing'
    var_10 = 'List'
    var_11 = 'MyList'
    var_12 = 1
    var_13 = 'test.sub_module'
    var_14 = 'utils'
    var_15 = 'helper'
    var_16 = 2
    var_17 = 'test.sub.sub_module'



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'not a name'
    var_5 = 'test_name'
    var_6 = 'test.alias_name'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = 'alias_name'



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = module_0.Parser()
    var_6 = 'os.path'
    var_7 = 'join'
    var_8 = 0
    var_9 = module_0.Parser()
    var_10 = 'sibling'
    var_11 = 'func'
    var_12 = 1
    var_13 = module_0.Parser()
    var_14 = 'collections'
    var_15 = 'defaultdict'
    var_16 = 'dd'



# Parsed testcases at query #24
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
    var_23 = 'z'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = [var_25]
    var_27 = 3
    var_28 = module_1.Constant()
    var_29 = module_1.Assign()
    var_30 = [var_29]
    var_31 = module_1.If()
    var_32 = [var_31]
    var_33 = module_0.walk_body(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 0
    var_37 = var_34[var_36]
    var_38 = var_34[var_13]
    var_39 = 'a'
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = [var_41]
    var_43 = 4
    var_44 = module_1.Constant()
    var_45 = module_1.Assign()
    var_46 = [var_45]
    var_47 = []
    var_48 = 'b'
    var_49 = module_1.Load()
    var_50 = module_1.Name()
    var_51 = [var_50]
    var_52 = 5
    var_53 = module_1.Constant()
    var_54 = module_1.Assign()
    var_55 = [var_54]
    var_56 = 'c'
    var_57 = module_1.Load()
    var_58 = module_1.Name()
    var_59 = [var_58]
    var_60 = 6
    var_61 = module_1.Constant()
    var_62 = module_1.Assign()
    var_63 = [var_62]
    var_64 = module_1.Try()
    var_65 = [var_64]
    var_66 = module_0.walk_body(var_65)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 3
    var_69 = True
    var_70 = module_1.Constant()
    var_71 = 'd'
    var_72 = module_1.Load()
    var_73 = module_1.Name()
    var_74 = [var_73]
    var_75 = 7
    var_76 = module_1.Constant()
    var_77 = module_1.Assign()
    var_78 = [var_77]
    var_79 = []
    var_80 = []
    var_81 = []
    var_82 = module_1.Try()
    var_83 = [var_82]
    var_84 = []
    var_85 = module_1.If()
    var_86 = [var_85]
    var_87 = module_0.walk_body(var_86)
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 1
    var_90 = var_88[var_36]



# Parsed testcases at query #25
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.alias'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 42
    var_6 = 'invalid expression'
    var_7 = 'alias'
    var_8 = 'test_module.alias.attr'
    var_9 = 'str'
    var_10 = {var_8: var_9}
    var_11 = module_0.Resolver(var_0, var_10)
    var_12 = 'alias.attr'



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = module_1.Load()
    var_8 = 'test.other'
    var_9 = 'str'
    var_10 = {var_8: var_9}
    var_11 = module_0.Resolver(var_0, var_10, var_4)
    var_12 = module_1.Load()
    var_13 = {}
    var_14 = 'self_ty'
    var_15 = module_0.Resolver(var_0, var_13, var_14)
    var_16 = module_1.Load()
    var_17 = 'test.TypeVar'
    var_18 = 'typing.TypeVar'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19, var_4)
    var_21 = 'TypeVar'
    var_22 = module_1.Load()



# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = 'os.path'
    var_7 = 'join'
    var_8 = 0
    var_9 = 'sibling'
    var_10 = 'func'
    var_11 = 'f'
    var_12 = 1



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = module_0.Parser()
    var_5 = "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"
    var_6 = var_4.parse(var_2, var_5)
    var_7 = module_0.Parser()
    var_8 = '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'
    var_9 = var_7.parse(var_2, var_8)
    var_10 = module_0.Parser()
    var_11 = "\nclass Outer:\n    class Inner:\n        def method(self):\n            '''Inner method.'''\n            pass\n    def outer_method(self):\n        '''Outer method.'''\n        pass\n"
    var_12 = var_10.parse(var_2, var_11)
    var_13 = module_0.Parser()
    var_14 = "\n@decorator1\n@decorator2\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_15 = var_13.parse(var_2, var_14)



# Parsed testcases at query #32
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'not a valid name'
    var_7 = 'test.name'
    var_8 = 'test.alias'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #33
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = 'other'
    var_10 = module_1.Load()
    var_11 = 'attr'
    var_12 = module_1.Load()
    var_13 = {}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = 42
    var_16 = module_1.Load()



# Parsed testcases at query #34
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
    var_30 = 'SomeType'
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()



# Parsed testcases at query #35
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
    var_14 = 'Bases\n-----\nBaseClass\n\nMembers\n-------\nattr1\nint\n\nattr2\nfloat\n\n'



# Parsed testcases at query #36
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = 'Union[int, str]'
    var_23 = {var_1: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()



# Parsed testcases at query #37
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 42
    var_6 = 'invalid name'
    var_7 = 'foo'
    var_8 = {}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #38
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
    var_39 = [var_9, var_38, var_16]
    var_40 = 0
    var_41 = var_38.body[var_40]
    var_42 = var_38.orelse[var_40]
    var_43 = [var_9, var_41, var_42, var_16]
    var_44 = module_0.walk_body(var_39)
    var_45 = list(var_44)
    var_46 = 'z'
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = [var_48]
    var_50 = 3
    var_51 = module_1.Constant()
    var_52 = module_1.Assign()
    var_53 = [var_52]
    var_54 = []
    var_55 = 'log'
    var_56 = module_1.Load()
    var_57 = module_1.Name()
    var_58 = []
    var_59 = []
    var_60 = module_1.Call(*var_58)
    var_61 = module_1.Expr()
    var_62 = [var_61]
    var_63 = 'cleanup'
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = []
    var_67 = []
    var_68 = module_1.Call(*var_66)
    var_69 = module_1.Expr()
    var_70 = [var_69]
    var_71 = module_1.Try()
    var_72 = [var_9, var_71, var_16]
    var_73 = var_71.body[var_40]
    var_74 = var_71.orelse[var_40]
    var_75 = var_71.finalbody[var_40]
    var_76 = [var_9, var_73, var_74, var_75, var_16]
    var_77 = module_0.walk_body(var_72)
    var_78 = list(var_77)
    var_79 = True
    var_80 = module_1.Constant()
    var_81 = [var_71]
    var_82 = []
    var_83 = module_1.If()
    var_84 = [var_83]
    var_85 = var_71.body[var_40]
    var_86 = var_71.orelse[var_40]
    var_87 = var_71.finalbody[var_40]
    var_88 = [var_85, var_86, var_87]
    var_89 = module_0.walk_body(var_84)
    var_90 = list(var_89)



# Parsed testcases at query #39
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'typing'
    var_6 = module_1.Load()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = 'other.Module'
    var_10 = 'module'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'other'
    var_14 = module_1.Load()
    var_15 = 'Module'
    var_16 = module_1.Load()
    var_17 = {}
    var_18 = module_0.Resolver(var_0, var_17)
    var_19 = 1
    var_20 = 'attr'
    var_21 = module_1.Load()



# Parsed testcases at query #40
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'not a name'
    var_5 = 'some_name'
    var_6 = 'root.some_name'
    var_7 = 'alias_name'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = {}
    var_11 = 'Self'
    var_12 = module_0.Resolver(var_0, var_10, var_11)



# Parsed testcases at query #41
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = 'collections'
    var_6 = 'defaultdict'
    var_7 = 0
    var_8 = 'os.path'
    var_9 = 'join'
    var_10 = 1
    var_11 = 'typing'
    var_12 = 'List'
    var_13 = 'list'



# Parsed testcases at query #42
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = 'invalid name'
    var_5 = 'valid_name'
    var_6 = 'test.nested'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = 'nested'



# Parsed testcases at query #43
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'test'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'invalid name'
    var_5 = {}
    var_6 = module_0.Resolver(var_1, var_5)
    var_7 = 'valid_name'
    var_8 = 'test.valid_name'
    var_9 = 'new_name'
    var_10 = {var_8: var_9}
    var_11 = module_0.Resolver(var_1, var_10)
    var_12 = 'nested.name'
    var_13 = 'test.nested.name'
    var_14 = 'new_nested_name'
    var_15 = {var_13: var_14}
    var_16 = module_0.Resolver(var_1, var_15)



# Parsed testcases at query #44
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'test_module.test_function'
    var_4 = '# Module `test_module`'
    var_5 = '# class TestClass'
    var_6 = '# test_function()'
    var_7 = 'Module docstring'
    var_8 = type('TestClass', (), {'__doc__': 'Class docstring'})
    var_9 = lambda : None



# Parsed testcases at query #45
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'SelfType'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test.TypeVar'
    var_12 = 'typing.TypeVar'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = 'TypeVar'
    var_16 = module_1.Load()
    var_17 = 'test.other'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = module_1.Load()
    var_22 = 'test.nested'
    var_23 = 'final'
    var_24 = {var_22: var_17, var_17: var_23}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = 'nested'
    var_27 = module_1.Load()



# Parsed testcases at query #46
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'not a valid name'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = 'collections'
    var_6 = 'defaultdict'
    var_7 = 0
    var_8 = 'os.path'
    var_9 = 'join'
    var_10 = 1
    var_11 = 'typing'
    var_12 = 'List'
    var_13 = 'list'



# Parsed testcases at query #49
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'CONSTANT'
    var_4 = module_0.Load()
    var_5 = 'int'
    var_6 = module_0.Load()
    var_7 = 42
    var_8 = 'VAR'
    var_9 = module_0.Load()
    var_10 = 'str'
    var_11 = module_0.Load()
    var_12 = None
    var_13 = 'ANOTHER_CONST'
    var_14 = module_0.Load()
    var_15 = 3.14
    var_16 = 'float'
    var_17 = 'INFERRED'
    var_18 = module_0.Load()
    var_19 = 2
    var_20 = module_0.Load()
    var_21 = '__all__'
    var_22 = module_0.Load()
    var_23 = 'public_func'
    var_24 = 'PublicClass'
    var_25 = module_0.Load()
    var_26 = 'variable'
    var_27 = module_0.Load()
    var_28 = 'test'
    var_29 = 'a'
    var_30 = module_0.Load()
    var_31 = 'b'
    var_32 = module_0.Load()



# Parsed testcases at query #50
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
    var_9 = 'test.typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Load()
    var_14 = 'str'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'test.typing.Optional'
    var_18 = 'Optional'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'test.typing.Custom'
    var_24 = 'Custom'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = module_1.Load()
    var_28 = module_1.Load()



# Parsed testcases at query #51
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'not a valid name'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = {}
    var_13 = module_0.Resolver(var_0, var_12)



# Parsed testcases at query #52
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
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = True
    var_14 = module_0.Constant()
    var_15 = 'y'
    var_16 = module_0.Load()
    var_17 = module_0.Name()
    var_18 = [var_17]
    var_19 = 2
    var_20 = module_0.Constant()
    var_21 = module_0.Assign()
    var_22 = [var_21]
    var_23 = 'z'
    var_24 = module_0.Load()
    var_25 = module_0.Name()
    var_26 = [var_25]
    var_27 = 3
    var_28 = module_0.Constant()
    var_29 = module_0.Assign()
    var_30 = [var_29]
    var_31 = module_0.If()
    var_32 = [var_31]
    var_33 = module_1.walk_body(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'a'
    var_37 = module_0.Load()
    var_38 = module_0.Name()
    var_39 = [var_38]
    var_40 = 4
    var_41 = module_0.Constant()
    var_42 = module_0.Assign()
    var_43 = [var_42]
    var_44 = []
    var_45 = 'b'
    var_46 = module_0.Load()
    var_47 = module_0.Name()
    var_48 = [var_47]
    var_49 = 5
    var_50 = module_0.Constant()
    var_51 = module_0.Assign()
    var_52 = [var_51]
    var_53 = 'c'
    var_54 = module_0.Load()
    var_55 = module_0.Name()
    var_56 = [var_55]
    var_57 = 6
    var_58 = module_0.Constant()
    var_59 = module_0.Assign()
    var_60 = [var_59]
    var_61 = module_0.Try()
    var_62 = [var_61]
    var_63 = module_1.walk_body(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 3
    var_66 = True
    var_67 = module_0.Constant()
    var_68 = False
    var_69 = module_0.Constant()
    var_70 = 'd'
    var_71 = module_0.Load()
    var_72 = module_0.Name()
    var_73 = [var_72]
    var_74 = 7
    var_75 = module_0.Constant()
    var_76 = module_0.Assign()
    var_77 = [var_76]
    var_78 = []
    var_79 = module_0.If()
    var_80 = [var_79]
    var_81 = []
    var_82 = module_0.If()
    var_83 = [var_82]
    var_84 = module_1.walk_body(var_83)
    var_85 = list(var_84)
    var_86 = len(var_85)
    assert var_86 == 1
    var_87 = var_85[var_68]
    var_88 = []
    var_89 = module_1.walk_body(var_88)
    var_90 = list(var_89)
    var_91 = len(var_90)
    assert var_91 == 0



# Parsed testcases at query #53
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 123
    var_5 = 'invalid name'
    var_6 = 'valid_name'
    var_7 = 'test.valid_name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9, var_2)



# Parsed testcases at query #54
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
    var_9 = 'test.typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Load()
    var_14 = 'str'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'test.typing.Optional'
    var_18 = 'Optional'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = {}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = 'SomeType'
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #55
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'test_module'
    var_4 = 'numpy'
    var_5 = 'np'
    var_6 = 'collections'
    var_7 = 'defaultdict'
    var_8 = 0
    var_9 = 'os.path'
    var_10 = 'join'
    var_11 = 'path_join'
    var_12 = 1



# Parsed testcases at query #56
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = module_0.Parser()
    var_7 = 'test_module'
    var_8 = 'TestClass'
    var_9 = 0
    var_10 = 'BaseClass'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_9]
    var_13 = var_12.value
    var_14 = [var_13]
    var_15 = []
    var_16 = var_6.class_api(var_7, var_8, var_14, var_15)
    var_17 = module_0.Parser()
    var_18 = 'test_module'
    var_19 = 'TestEnum'
    var_20 = 'enum.Enum'
    var_21 = module_1.parse(var_20)
    var_22 = var_21.body[var_9]
    var_23 = var_22.value
    var_24 = [var_23]
    var_25 = 'MEMBER1'
    var_26 = module_1.Load()
    var_27 = 'int'
    var_28 = module_1.Load()
    var_29 = None
    var_30 = 'MEMBER2'
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = var_17.class_api(var_18, var_19, var_24, var_15)
    var_34 = module_0.Parser()
    var_35 = 'test_module'
    var_36 = 'TestClass'
    var_37 = []
    var_38 = 'public_attr'
    var_39 = module_1.Load()
    var_40 = module_1.Load()
    var_41 = '_private_attr'
    var_42 = module_1.Load()
    var_43 = module_1.Load()
    var_44 = var_34.class_api(var_35, var_36, var_37, var_15)
    var_45 = module_0.Parser()
    var_46 = 'test_module'
    var_47 = 'TestClass'
    var_48 = []
    var_49 = 'attr1'
    var_50 = module_1.Load()
    var_51 = module_1.Load()
    var_52 = var_45.class_api(var_46, var_47, var_48, var_15)



# Parsed testcases at query #57
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
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #58
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
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = module_1.Expr()
    var_13 = [var_9, var_12]
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = 'y'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = [var_20]
    var_22 = 2
    var_23 = module_1.Constant()
    var_24 = module_1.Assign()
    var_25 = [var_24]
    var_26 = 'world'
    var_27 = module_1.Constant()
    var_28 = module_1.Expr()
    var_29 = [var_28]
    var_30 = module_1.If()
    var_31 = [var_30]
    var_32 = module_0.walk_body(var_31)
    var_33 = list(var_32)
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = [var_35]
    var_37 = module_1.Constant()
    var_38 = module_1.Assign()
    var_39 = module_1.Constant()
    var_40 = module_1.Expr()
    var_41 = [var_38, var_40]
    var_42 = 'z'
    var_43 = module_1.Load()
    var_44 = module_1.Name()
    var_45 = [var_44]
    var_46 = 3
    var_47 = module_1.Constant()
    var_48 = module_1.Assign()
    var_49 = [var_48]
    var_50 = []
    var_51 = 'try_else'
    var_52 = module_1.Constant()
    var_53 = module_1.Expr()
    var_54 = [var_53]
    var_55 = 'finally'
    var_56 = module_1.Constant()
    var_57 = module_1.Expr()
    var_58 = [var_57]
    var_59 = module_1.Try()
    var_60 = [var_59]
    var_61 = module_0.walk_body(var_60)
    var_62 = list(var_61)
    var_63 = module_1.Load()
    var_64 = module_1.Name()
    var_65 = [var_64]
    var_66 = module_1.Constant()
    var_67 = module_1.Assign()
    var_68 = module_1.Constant()
    var_69 = module_1.Expr()
    var_70 = module_1.Constant()
    var_71 = module_1.Expr()
    var_72 = [var_67, var_69, var_71]
    var_73 = True
    var_74 = module_1.Constant()
    var_75 = False
    var_76 = module_1.Constant()
    var_77 = 'nested'
    var_78 = module_1.Constant()
    var_79 = module_1.Expr()
    var_80 = [var_79]
    var_81 = []
    var_82 = module_1.If()
    var_83 = [var_82]
    var_84 = []
    var_85 = module_1.If()
    var_86 = [var_85]
    var_87 = module_0.walk_body(var_86)
    var_88 = list(var_87)
    var_89 = module_1.Constant()
    var_90 = module_1.Expr()
    var_91 = [var_90]
    var_92 = 'try_body'
    var_93 = module_1.Constant()
    var_94 = module_1.Expr()
    var_95 = [var_94]
    var_96 = 'type'
    var_97 = 'name'
    var_98 = 'body'
    var_99 = 'Exception'
    var_100 = module_1.Load()
    var_101 = module_1.Name()
    var_102 = 'e'
    var_103 = 'handler'
    var_104 = module_1.Constant()
    var_105 = module_1.Expr()
    var_106 = [var_105]
    var_107 = {var_96: var_101, var_97: var_102, var_98: var_106}
    var_108 = [var_107]
    var_109 = []
    var_110 = []
    var_111 = module_1.Try()
    var_112 = [var_111]
    var_113 = module_0.walk_body(var_112)
    var_114 = list(var_113)
    var_115 = module_1.Constant()
    var_116 = module_1.Expr()
    var_117 = module_1.Constant()
    var_118 = module_1.Expr()
    var_119 = [var_116, var_118]



# Parsed testcases at query #59
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = 'not a name'
    var_5 = 'test_name'
    var_6 = 'test.alias_name'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = 'alias_name'



# Parsed testcases at query #60
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.typing.Union'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    var_4 = 'Union'
    var_5 = module_1.Load()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Optional'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = 'test.typing.List'
    var_17 = {var_16: var_16}
    var_18 = module_0.Resolver(var_0, var_17)
    var_19 = 'List'
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'CustomType'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = module_1.Load()



# Parsed testcases at query #61
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = 'module.replacement'
    var_23 = 'final'
    var_24 = {var_1: var_2, var_22: var_23}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = module_1.Load()



# Parsed testcases at query #62
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'
    var_3 = 1
    var_4 = module_0.Constant()
    var_5 = 2
    var_6 = module_0.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_0.Load()
    var_9 = module_0.Tuple()
    var_10 = module_1.const_type(var_9)
    assert var_10 == 'tuple[int, int]'
    var_11 = module_0.Constant()
    var_12 = module_0.Constant()
    var_13 = [var_11, var_12]
    var_14 = module_0.Load()
    var_15 = module_0.List()
    var_16 = module_1.const_type(var_15)
    assert var_16 == 'list[int, int]'
    var_17 = module_0.Constant()
    var_18 = module_0.Constant()
    var_19 = [var_17, var_18]
    var_20 = module_0.Set()
    var_21 = module_1.const_type(var_20)
    assert var_21 == 'set[int, int]'
    var_22 = 'a'
    var_23 = module_0.Constant()
    var_24 = 'b'
    var_25 = module_0.Constant()
    var_26 = [var_23, var_25]
    var_27 = module_0.Constant()
    var_28 = module_0.Constant()
    var_29 = [var_27, var_28]
    var_30 = module_0.Dict()
    var_31 = module_1.const_type(var_30)
    assert var_31 == 'dict[str, int]'
    var_32 = 'int'
    var_33 = module_0.Load()
    var_34 = module_0.Name()
    var_35 = []
    var_36 = []
    var_37 = module_0.Call(*var_35)
    var_38 = module_1.const_type(var_37)
    assert var_38 == 'int'
    var_39 = 'typing'
    var_40 = module_0.Load()
    var_41 = module_0.Name()
    var_42 = 'List'
    var_43 = module_0.Load()
    var_44 = module_0.Attribute()
    var_45 = []
    var_46 = []
    var_47 = module_0.Call(*var_45)
    var_48 = module_1.const_type(var_47)
    assert var_48 == 'Any'
    var_49 = module_0.Constant()
    var_50 = module_0.BitOr()
    var_51 = module_0.Constant()
    var_52 = module_0.BinOp()
    var_53 = module_1.const_type(var_52)
    assert var_53 == 'Any'
    var_54 = []
    var_55 = module_0.Load()
    var_56 = module_0.List()
    var_57 = module_1.const_type(var_56)
    assert var_57 == 'Any'
    var_58 = module_0.Constant()
    var_59 = module_0.Constant()
    var_60 = [var_58, var_59]
    var_61 = module_0.Load()
    var_62 = module_0.List()
    var_63 = module_1.const_type(var_62)
    assert var_63 == 'Any'



# Parsed testcases at query #63
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
    var_7 = 'other_module'
    var_8 = module_1.Load()
    var_9 = 'SomeClass'
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = module_1.Load()
    var_13 = 'append'
    var_14 = module_1.Load()



# Parsed testcases at query #64
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.typing'
    var_2 = 'typing'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = {}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = 'other_module'
    var_11 = module_1.Load()
    var_12 = 'SomeClass'
    var_13 = module_1.Load()
    var_14 = {}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = 42
    var_17 = 'attr'
    var_18 = module_1.Load()



# Parsed testcases at query #65
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.submodule'
    var_2 = 'module.submodule.function'
    var_3 = 'Documentation for submodule'
    var_4 = 'Documentation for function'
    var_5 = 'module'
    var_6 = "This is the module's docstring"
    var_7 = module_0.doctest(var_6)
    var_8 = "This is the submodule's docstring"
    var_9 = module_0.doctest(var_8)
    var_10 = "This is the function's docstring"
    var_11 = module_0.doctest(var_10)



# Parsed testcases at query #66
#--------------------------




# Parsed testcases at query #67
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.typing.Union'
    var_2 = 'test_module.typing.Optional'
    var_3 = 'test_module.typing.List'
    var_4 = 'Union'
    var_5 = 'Optional'
    var_6 = 'list'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = module_1.Load()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = 'str'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'List'
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'Other'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = module_1.Load()



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = 'typing.Dict'
    var_29 = 'Dict'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_0, var_30)
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()
    var_35 = module_1.Load()
    var_36 = module_1.Load()



# Parsed testcases at query #70
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.A'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 42
    var_6 = 'not a name'
    var_7 = 'A'
    var_8 = 'A.B'



# Parsed testcases at query #71
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid expression'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = 'test.module.name'
    var_13 = 'str'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = 'module.name'
    var_17 = 'test.List'
    var_18 = 'list'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = 'List[int]'



# Parsed testcases at query #72
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid expression'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = 'test.attr'
    var_13 = 'str'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = 'attr'



# Parsed testcases at query #73
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = module_0.Parser()
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'BaseClass'
    var_10 = []
    var_11 = var_6.class_api(var_7, var_8, var_3, var_10)
    var_12 = module_0.Parser()
    var_13 = 'test_module'
    var_14 = 'test_module.TestClass'
    var_15 = []
    var_16 = 'attr1'
    var_17 = module_1.Load()
    var_18 = 'int'
    var_19 = module_1.Load()
    var_20 = None
    var_21 = var_12.class_api(var_13, var_14, var_15, var_10)
    var_22 = module_0.Parser()
    var_23 = 'test_module'
    var_24 = 'test_module.TestClass'
    var_25 = []
    var_26 = module_1.Load()
    var_27 = 1
    var_28 = var_22.class_api(var_23, var_24, var_25, var_10)
    var_29 = module_0.Parser()
    var_30 = 'test_module'
    var_31 = 'test_module.TestClass'
    var_32 = []
    var_33 = module_1.Load()
    var_34 = var_29.class_api(var_30, var_31, var_32, var_10)
    var_35 = module_0.Parser()
    var_36 = 'test_module'
    var_37 = 'test_module.TestClass'
    var_38 = 'enum.Enum'
    var_39 = 'ATTR1'
    var_40 = module_1.Load()
    var_41 = module_1.Load()
    var_42 = var_35.class_api(var_36, var_37, var_32, var_10)
    var_43 = module_0.Parser()
    var_44 = 'test_module'
    var_45 = 'test_module.TestClass'
    var_46 = module_1.Load()
    var_47 = var_43.class_api(var_44, var_45, var_32, var_10)
    var_48 = module_0.Parser()
    var_49 = 'test_module'
    var_50 = 'test_module.TestClass'
    var_51 = module_1.Load()
    var_52 = var_48.class_api(var_49, var_50, var_32, var_10)



# Parsed testcases at query #74
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'List'
    var_6 = module_1.Load()
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'test_module.typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'test_module.typing.Optional'
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
    var_35 = {var_11: var_12}
    var_36 = module_0.Resolver(var_0, var_35)
    var_37 = module_1.Load()
    var_38 = module_1.Load()
    var_39 = module_1.Load()



# Parsed testcases at query #75
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
    var_19 = module_0.Load()
    var_20 = module_0.Tuple()
    var_21 = module_1.const_type(var_20)
    assert var_21 == 'tuple[int, int]'
    var_22 = module_0.Constant()
    var_23 = module_0.Constant()
    var_24 = [var_22, var_23]
    var_25 = module_0.Load()
    var_26 = module_0.List()
    var_27 = module_1.const_type(var_26)
    assert var_27 == 'list[int, int]'
    var_28 = module_0.Constant()
    var_29 = module_0.Constant()
    var_30 = [var_28, var_29]
    var_31 = module_0.Set()
    var_32 = module_1.const_type(var_31)
    assert var_32 == 'set[int, int]'
    var_33 = 'a'
    var_34 = module_0.Constant()
    var_35 = 'b'
    var_36 = module_0.Constant()
    var_37 = [var_34, var_36]
    var_38 = module_0.Constant()
    var_39 = module_0.Constant()
    var_40 = [var_38, var_39]
    var_41 = module_0.Dict()
    var_42 = module_1.const_type(var_41)
    assert var_42 == 'dict[str, int]'
    var_43 = 'int'
    var_44 = module_0.Load()
    var_45 = module_0.Name()
    var_46 = []
    var_47 = []
    var_48 = module_0.Call(*var_46)
    var_49 = module_1.const_type(var_48)
    assert var_49 == 'int'
    var_50 = 'x'
    var_51 = module_0.Load()
    var_52 = module_0.Name()
    var_53 = module_1.const_type(var_52)
    assert var_53 == 'Any'
    var_54 = module_0.Constant()
    var_55 = module_0.Constant()



# Parsed testcases at query #76
#--------------------------




# Parsed testcases at query #77
#--------------------------




# Parsed testcases at query #78
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replaced_name'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test_module.other'
    var_12 = 'other_name'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'test_module.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = 'test_module.A'
    var_23 = 'test_module.B'
    var_24 = 'final_name'
    var_25 = {var_22: var_23, var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = 'A'
    var_28 = module_1.Load()



# Parsed testcases at query #79
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.SomeClass'
    var_2 = 'SomeClass'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 42
    var_6 = 'not a valid name!'



# Parsed testcases at query #80
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
    var_6 = 1
    var_7 = None
    var_8 = 'attr2'
    var_9 = 'hello'
    var_10 = 'attr3'
    var_11 = 3.14



# Parsed testcases at query #81
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
    var_12 = module_0.Constant()
    var_13 = 2
    var_14 = module_0.Constant()
    var_15 = [var_12, var_14]
    var_16 = module_0.Load()
    var_17 = module_0.Tuple()
    var_18 = module_1.const_type(var_17)
    assert var_18 == 'tuple[int, int]'
    var_19 = module_0.Constant()
    var_20 = module_0.Constant()
    var_21 = [var_19, var_20]
    var_22 = module_0.Load()
    var_23 = module_0.List()
    var_24 = module_1.const_type(var_23)
    assert var_24 == 'list[int, int]'
    var_25 = module_0.Constant()
    var_26 = module_0.Constant()
    var_27 = [var_25, var_26]
    var_28 = module_0.Set()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'set[int, int]'
    var_30 = 'a'
    var_31 = module_0.Constant()
    var_32 = 'b'
    var_33 = module_0.Constant()
    var_34 = [var_31, var_33]
    var_35 = module_0.Constant()
    var_36 = module_0.Constant()
    var_37 = [var_35, var_36]
    var_38 = module_0.Dict()
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'dict[str, int]'
    var_40 = 'int'
    var_41 = module_0.Load()
    var_42 = module_0.Name()
    var_43 = []
    var_44 = []
    var_45 = module_0.Call(*var_43)
    var_46 = module_1.const_type(var_45)
    assert var_46 == 'int'
    var_47 = 'typing'
    var_48 = module_0.Load()
    var_49 = module_0.Name()
    var_50 = 'List'
    var_51 = module_0.Load()
    var_52 = module_0.Attribute()
    var_53 = []
    var_54 = []
    var_55 = module_0.Call(*var_53)
    var_56 = module_1.const_type(var_55)
    assert var_56 == 'list'
    var_57 = module_0.Constant()
    var_58 = module_0.Constant()
    var_59 = None
    var_60 = module_1.const_type(var_59)
    assert var_60 == 'Any'



# Parsed testcases at query #82
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'expression'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_expression'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = {var_1: var_11, var_11: var_2}
    var_23 = module_0.Resolver(var_0, var_22)
    var_24 = module_1.Load()



# Parsed testcases at query #83
#--------------------------




# Parsed testcases at query #84
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.func'
    var_3 = 'test_module.Class'
    var_4 = '# Module `test_module`'
    var_5 = '# func()'
    var_6 = '# class Class'
    var_7 = 'Module docstring'
    var_8 = lambda : None
    var_9 = type('Class', (), {'__doc__': 'Class docstring'})
    var_10 = 'Module docstring'
    var_11 = module_0.doctest(var_10)
    var_12 = 'Function docstring'
    var_13 = module_0.doctest(var_12)
    var_14 = 'Class docstring'
    var_15 = module_0.doctest(var_14)



# Parsed testcases at query #85
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.typing.Union'
    var_2 = 'test.typing.Optional'
    var_3 = 'test.typing.List'
    var_4 = 'Union'
    var_5 = 'Optional'
    var_6 = 'list'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = module_1.Load()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = 'str'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'List'
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'Dict'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = module_1.Load()



# Parsed testcases at query #86
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = module_0.Parser()
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'BaseClass'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_11]
    var_13 = []
    var_14 = var_6.class_api(var_7, var_8, var_12, var_13)
    var_15 = module_0.Parser()
    var_16 = 'test_module'
    var_17 = 'test_module.TestClass'
    var_18 = []
    var_19 = 'public_attr'
    var_20 = 'int'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = None
    var_24 = var_15.class_api(var_16, var_17, var_18, var_13)
    var_25 = module_0.Parser()
    var_26 = 'test_module'
    var_27 = 'test_module.TestEnum'
    var_28 = 'Enum'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = [var_30]
    var_32 = 'VALUE1'
    var_33 = module_1.Load()
    var_34 = module_1.Name()
    var_35 = var_25.class_api(var_26, var_27, var_31, var_13)
    var_36 = module_0.Parser()
    var_37 = 'test_module'
    var_38 = 'test_module.TestClass'
    var_39 = []
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = var_36.class_api(var_37, var_38, var_39, var_13)
    var_43 = module_0.Parser()
    var_44 = 'test_module'
    var_45 = 'test_module.TestClass'
    var_46 = []
    var_47 = 42
    var_48 = module_1.Constant()
    var_49 = var_43.class_api(var_44, var_45, var_46, var_13)



# Parsed testcases at query #87
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid expression'
    var_7 = 'test.name'
    var_8 = 'int'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = 'test.module.Class'
    var_13 = 'str'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = 'module.Class'



# Parsed testcases at query #88
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
    var_6 = 1
    var_7 = None
    var_8 = 'attr2'
    var_9 = 'hello'
    var_10 = 'attr3'
    var_11 = 3.14



# Parsed testcases at query #89
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = 'return'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = 'module'
    var_8 = False
    var_9 = 'self'
    var_10 = 'module.Class'
    var_11 = module_1.Load()
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = True
    var_15 = 'cls'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = '*'
    var_21 = None
    var_22 = module_1.Load()



# Parsed testcases at query #90
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = 'not a name'
    var_5 = 'valid_name'
    var_6 = 'test.valid_name'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = module_0.Resolver(var_0, var_8)



# Parsed testcases at query #91
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
    var_18 = 'cls'
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = '*'
    var_24 = None
    var_25 = module_1.Load()
    var_26 = '**'
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #92
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = 'not a valid expression'
    var_5 = 'test.name'
    var_6 = 'int'
    var_7 = {var_5: var_6}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = 'name'
    var_10 = 'test.module.Class'
    var_11 = {var_10: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = 'module.Class'



# Parsed testcases at query #93
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = 'system'
    var_5 = 'collections'
    var_6 = 'defaultdict'
    var_7 = 0
    var_8 = 'os.path'
    var_9 = 'join'
    var_10 = 1



# Parsed testcases at query #94
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
    var_23 = {}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = 'SomeClass'
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #95
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = module_0.ClassDef()
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = var_7.bases
    var_11 = var_7.body
    var_12 = 'TestClassWithBases'
    var_13 = 'BaseClass'
    var_14 = module_0.Load()
    var_15 = module_0.Name()
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = module_0.ClassDef()
    var_21 = 'test_module.TestClassWithBases'
    var_22 = var_20.bases
    var_23 = var_20.body
    var_24 = 'TestClassWithMembers'
    var_25 = []
    var_26 = []
    var_27 = 'member1'
    var_28 = 'int'
    var_29 = module_0.Load()
    var_30 = module_0.Name()
    var_31 = None
    var_32 = 'member2'
    var_33 = 'str'
    var_34 = module_0.Load()
    var_35 = module_0.Name()
    var_36 = []
    var_37 = 'test_module.TestClassWithMembers'
    var_38 = 'TestEnum'
    var_39 = 'Enum'
    var_40 = module_0.Load()
    var_41 = module_0.Name()
    var_42 = [var_41]
    var_43 = []
    var_44 = 'VALUE1'
    var_45 = module_0.Constant()
    var_46 = 'VALUE2'
    var_47 = 2
    var_48 = module_0.Constant()
    var_49 = []
    var_50 = 'test_module.TestEnum'
    var_51 = 'TestClassWithDeleted'
    var_52 = []
    var_53 = []
    var_54 = module_0.Load()
    var_55 = module_0.Name()
    var_56 = []
    var_57 = 'test_module.TestClassWithDeleted'



# Parsed testcases at query #96
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'alias.name'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = module_1.Load()
    var_10 = {}
    var_11 = 'SelfType'
    var_12 = module_0.Resolver(var_0, var_10, var_11)
    var_13 = module_1.Load()
    var_14 = 'alias.other'
    var_15 = 'final.name'
    var_16 = {var_1: var_14, var_14: var_15}
    var_17 = module_0.Resolver(var_0, var_16)
    var_18 = module_1.Load()
    var_19 = 'test.T'
    var_20 = "typing.TypeVar('T')"
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = 'T'
    var_24 = module_1.Load()



# Parsed testcases at query #97
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'typing'
    var_6 = module_1.Load()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = {var_1: var_2}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'other_module'
    var_12 = module_1.Load()
    var_13 = 'SomeClass'
    var_14 = module_1.Load()
    var_15 = {var_1: var_2}
    var_16 = module_0.Resolver(var_0, var_15)
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'nested'
    var_20 = module_1.Load()



# Parsed testcases at query #98
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
    var_9 = 'attr2'
    var_10 = 42
    var_11 = 'float'
    var_12 = 'attr3'
    var_13 = '# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id="test_module-testclass"></a>\n\nBases\n-----\nBaseClass\n\nMembers\n-------\nName\nType\nattr1\nint\nattr2\nfloat\n'



# Parsed testcases at query #99
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
    var_22 = [var_9]
    var_23 = [var_16]
    var_24 = module_1.If()
    var_25 = [var_24]
    var_26 = module_0.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = [var_9]
    var_29 = []
    var_30 = [var_16]
    var_31 = []
    var_32 = module_1.Try()
    var_33 = [var_32]
    var_34 = module_0.walk_body(var_33)
    var_35 = list(var_34)
    var_36 = True
    var_37 = module_1.Constant()
    var_38 = [var_24]
    var_39 = []
    var_40 = module_1.If()
    var_41 = [var_40]
    var_42 = module_0.walk_body(var_41)
    var_43 = list(var_42)
    var_44 = 'Exception'
    var_45 = module_1.Load()
    var_46 = module_1.Name()
    var_47 = None
    var_48 = [var_9]
    var_49 = [var_16]
    var_50 = []
    var_51 = []
    var_52 = [var_32]
    var_53 = module_0.walk_body(var_52)
    var_54 = list(var_53)
    var_55 = [var_9, var_24, var_16, var_32]
    var_56 = module_0.walk_body(var_55)
    var_57 = list(var_56)



# Parsed testcases at query #100
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'SelfType'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test_module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'test_module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'test_module.a'
    var_23 = 'test_module.b'
    var_24 = 'b'
    var_25 = 'c'
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.Resolver(var_0, var_26)
    var_28 = 'a'
    var_29 = module_1.Load()



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'module'
    var_5 = False
    var_6 = module_0.Parser()
    var_7 = 'self'
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = 'str'
    var_11 = module_1.Load()
    var_12 = True
    var_13 = module_0.Parser()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_0.Parser()
    var_17 = module_1.Load()
    var_18 = '*'
    var_19 = module_1.Load()
    var_20 = '**'
    var_21 = module_0.Parser()
    var_22 = module_1.Load()
    var_23 = 'z'
    var_24 = module_1.Load()



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = module_0.Parser()
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 0
    var_10 = 'BaseClass'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_9]
    var_13 = var_12.value
    var_14 = [var_13]
    var_15 = []
    var_16 = var_6.class_api(var_7, var_8, var_14, var_15)
    var_17 = module_0.Parser()
    var_18 = 'test_module'
    var_19 = 'test_module.TestClass'
    var_20 = []
    var_21 = 'attr1'
    var_22 = module_1.Load()
    var_23 = 'int'
    var_24 = module_1.Load()
    var_25 = None
    var_26 = 1
    var_27 = 'attr2'
    var_28 = module_1.Load()
    var_29 = 'str'
    var_30 = module_1.Load()
    var_31 = var_17.class_api(var_18, var_19, var_20, var_15)
    var_32 = module_0.Parser()
    var_33 = 'test_module'
    var_34 = 'test_module.TestClass'
    var_35 = []
    var_36 = 'hello'
    var_37 = var_32.class_api(var_33, var_34, var_35, var_15)
    var_38 = module_0.Parser()
    var_39 = 'test_module'
    var_40 = 'test_module.TestEnum'
    var_41 = 'enum.Enum'
    var_42 = module_1.parse(var_41)
    var_43 = var_42.body[var_9]
    var_44 = var_43.value
    var_45 = [var_44]
    var_46 = 'VALUE1'
    var_47 = module_1.Load()
    var_48 = 'VALUE2'
    var_49 = module_1.Load()
    var_50 = 2
    var_51 = var_38.class_api(var_39, var_40, var_45, var_15)
    var_52 = module_0.Parser()
    var_53 = 'test_module'
    var_54 = 'test_module.TestClass'
    var_55 = []
    var_56 = module_1.Load()
    var_57 = module_1.Load()
    var_58 = var_52.class_api(var_53, var_54, var_55, var_15)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 10
    var_7 = module_0.Parser()
    var_8 = 'y'
    var_9 = 'hello'
    var_10 = 'str'
    var_11 = module_0.Parser()
    var_12 = 'z'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = module_0.Parser()
    var_17 = '__all__'
    var_18 = 'func1'
    var_19 = 'func2'
    var_20 = module_0.Parser()
    var_21 = 'a'
    var_22 = 'b'
    var_23 = module_1.Load()
    var_24 = module_0.Parser()



# Parsed testcases at query #4
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
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 0
    var_44 = var_41[var_43]
    var_45 = var_41[var_20]
    var_46 = 'z'
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = [var_48]
    var_50 = 3
    var_51 = module_1.Constant()
    var_52 = module_1.Assign()
    var_53 = [var_52]
    var_54 = []
    var_55 = 'log'
    var_56 = module_1.Load()
    var_57 = module_1.Name()
    var_58 = []
    var_59 = []
    var_60 = module_1.Call(*var_58)
    var_61 = module_1.Expr()
    var_62 = [var_61]
    var_63 = 'cleanup'
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = []
    var_67 = []
    var_68 = module_1.Call(*var_66)
    var_69 = module_1.Expr()
    var_70 = [var_69]
    var_71 = module_1.Try()
    var_72 = [var_71]
    var_73 = module_0.walk_body(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 3
    var_76 = True
    var_77 = module_1.Constant()
    var_78 = False
    var_79 = module_1.Constant()
    var_80 = 'a'
    var_81 = module_1.Load()
    var_82 = module_1.Name()
    var_83 = [var_82]
    var_84 = 4
    var_85 = module_1.Constant()
    var_86 = module_1.Assign()
    var_87 = [var_86]
    var_88 = []
    var_89 = module_1.If()
    var_90 = [var_89]
    var_91 = []
    var_92 = module_1.If()
    var_93 = [var_92]
    var_94 = module_0.walk_body(var_93)
    var_95 = list(var_94)
    var_96 = len(var_95)
    assert var_96 == 1
    var_97 = var_95[var_78]
    var_98 = 'b'
    var_99 = module_1.Load()
    var_100 = module_1.Name()
    var_101 = [var_100]
    var_102 = 5
    var_103 = module_1.Constant()
    var_104 = module_1.Assign()
    var_105 = True
    var_106 = module_1.Constant()
    var_107 = 'func'
    var_108 = module_1.Load()
    var_109 = module_1.Name()
    var_110 = []
    var_111 = []
    var_112 = module_1.Call(*var_110)
    var_113 = module_1.Expr()
    var_114 = [var_113]
    var_115 = []
    var_116 = module_1.If()
    var_117 = []
    var_118 = []
    var_119 = 'c'
    var_120 = module_1.Load()
    var_121 = module_1.Name()
    var_122 = [var_121]
    var_123 = 6
    var_124 = module_1.Constant()
    var_125 = module_1.Assign()
    var_126 = [var_125]
    var_127 = []
    var_128 = module_1.Try()
    var_129 = [var_104, var_116, var_128]
    var_130 = module_0.walk_body(var_129)
    var_131 = list(var_130)
    var_132 = len(var_131)
    assert var_132 == 3



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = module_0.Resolver(var_0, var_7)
    var_9 = 'unknown_name'
    var_10 = module_1.Load()
    var_11 = {}
    var_12 = 'self'
    var_13 = module_0.Resolver(var_0, var_11, var_12)
    var_14 = module_1.Load()
    var_15 = 'test_module.TypeVar'
    var_16 = 'typing.TypeVar'
    var_17 = {var_15: var_16}
    var_18 = module_0.Resolver(var_0, var_17)
    var_19 = 'TypeVar'
    var_20 = module_1.Load()



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 123
    var_5 = {}
    var_6 = module_0.Resolver(var_0, var_5, var_2)
    var_7 = 'not_a_name'
    var_8 = {}
    var_9 = module_0.Resolver(var_0, var_8, var_2)
    var_10 = 'some_name'
    var_11 = 'root.some_name'
    var_12 = 'alias_name'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13, var_2)
    var_15 = 'root.TypeVar'
    var_16 = 'typing.TypeVar'
    var_17 = {var_15: var_16}
    var_18 = module_0.Resolver(var_0, var_17, var_2)
    var_19 = "TypeVar('T')"



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    '''Module docstring.'''\n    x = 1\n    y: int = 2\n    def foo(a: int) -> str:\n        '''Function docstring.'''\n        return str(a)\n    class Bar:\n        '''Class docstring.'''\n        z: float = 3.0\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = '\n    from typing import List\n    import os\n    x: List[int] = [1, 2, 3]\n    '
    var_5 = 'test_imports'
    var_6 = var_0.parse(var_5, var_4)
    var_7 = '\n    MyType = dict[str, int]\n    x: MyType = {"a": 1}\n    '
    var_8 = 'test_alias'
    var_9 = var_0.parse(var_8, var_7)
    var_10 = '\n    CONSTANT = 42\n    ANOTHER_CONST = "hello"\n    '
    var_11 = 'test_const'
    var_12 = var_0.parse(var_11, var_10)
    var_13 = '\n    __all__ = ["public_func", "PublicClass"]\n    def public_func():\n        pass\n    def _private_func():\n        pass\n    class PublicClass:\n        pass\n    class _PrivateClass:\n        pass\n    '
    var_14 = 'test_all'
    var_15 = var_0.parse(var_14, var_13)
    var_16 = '\n    def decorator(func):\n        return func\n    @decorator\n    def decorated_func():\n        pass\n    '
    var_17 = 'test_decorator'
    var_18 = var_0.parse(var_17, var_16)
    var_19 = 'test_decorator.decorated_func'
    var_20 = var_0.doc[var_19]
    var_21 = '\n    class Base:\n        pass\n    class Derived(Base):\n        pass\n    '
    var_22 = 'test_inheritance'
    var_23 = var_0.parse(var_22, var_21)
    var_24 = 'test_inheritance.Derived'
    var_25 = var_0.doc[var_24]
    var_26 = '\n    from enum import Enum\n    class Color(Enum):\n        RED = 1\n        GREEN = 2\n        BLUE = 3\n    '
    var_27 = 'test_enum'
    var_28 = var_0.parse(var_27, var_26)
    var_29 = 'test_enum.Color'
    var_30 = var_0.doc[var_29]
    var_31 = '\n    def complex_func(\n        a: int,\n        b: str = "default",\n        *args: float,\n        c: bool = True,\n        **kwargs: dict\n    ) -> list:\n        pass\n    '
    var_32 = 'test_annotations'
    var_33 = var_0.parse(var_32, var_31)
    var_34 = 'test_annotations.complex_func'
    var_35 = var_0.doc[var_34]
    var_36 = '\n    async def async_func():\n        pass\n    '
    var_37 = 'test_async'
    var_38 = var_0.parse(var_37, var_36)
    var_39 = 'test_async.async_func'
    var_40 = var_0.doc[var_39]
    var_41 = '\n    class Outer:\n        class Inner:\n            pass\n    '
    var_42 = 'test_nested'
    var_43 = var_0.parse(var_42, var_41)
    var_44 = '\n    class Test:\n        x: int = 1\n        del x\n    '
    var_45 = 'test_deletion'
    var_46 = var_0.parse(var_45, var_44)
    var_47 = 'test_deletion.Test'
    var_48 = var_0.doc[var_47]



# Parsed testcases at query #8
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
    var_10 = module_1.Constant()
    var_11 = module_0.Parser()
    var_12 = module_1.Constant()
    var_13 = module_0.Parser()
    var_14 = '__all__'
    var_15 = 'public_func'
    var_16 = module_1.Constant()
    var_17 = 'public_class'
    var_18 = module_1.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_1.List()
    var_21 = module_0.Parser()
    var_22 = 'a'
    var_23 = 'b'
    var_24 = module_1.Constant()
    var_25 = module_0.Parser()
    var_26 = module_1.Constant()
    var_27 = module_0.Parser()
    var_28 = 'some_func'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = 'test_module.test_var'
    var_32 = 'NOT_FOUND'



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = 'ANOTHER_CONST'
    var_9 = 3.14
    var_10 = 'float'
    var_11 = 'INFERRED_CONST'
    var_12 = 2
    var_13 = module_1.Load()
    var_14 = '__all__'
    var_15 = 'public_func'
    var_16 = 'PublicClass'
    var_17 = module_1.Load()
    var_18 = 'non_const'
    var_19 = 100
    var_20 = 'a'
    var_21 = 'b'



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
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



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Load()
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = 42
    var_6 = 'module'
    var_7 = 'y'
    var_8 = 3.14
    var_9 = 'float'
    var_10 = 'z'
    var_11 = 1
    var_12 = 2
    var_13 = '__all__'
    var_14 = 'public_func'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = module_1.Load()
    var_18 = 'c'
    var_19 = 'd'
    var_20 = 100



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a simple docstring.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a simple docstring.'
    var_2 = ">>> print('Hello, World!')"
    var_3 = "```python\n>>> print('Hello, World!')\n```"
    var_4 = module_0.doctest(var_2)
    var_5 = 'This is a docstring with a doctest:\n\n>>> x = 5\n>>> y = 10\n>>> x + y\n15\n\nAnd some more text.'



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 5
    var_7 = module_0.Parser()
    var_8 = 'y'
    var_9 = 'hello'
    var_10 = 'str'
    var_11 = module_0.Parser()
    var_12 = 'z'
    var_13 = 1
    var_14 = 2
    var_15 = module_0.Parser()
    var_16 = '__all__'
    var_17 = 'func1'
    var_18 = 'func2'
    var_19 = module_0.Parser()
    var_20 = 'non_const'
    var_21 = 42
    var_22 = module_0.Parser()
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 10



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
    var_24 = 'Dict'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = module_1.Load()
    var_28 = module_1.Load()



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = {}
    var_5 = module_0.Resolver(var_0, var_4)
    var_6 = 'invalid expression'
    var_7 = 'test.name'
    var_8 = 'new_name'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = 'name'
    var_12 = 'test.obj.attr'
    var_13 = 'new_attr'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = 'obj.attr'



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = 'root.other'
    var_8 = {var_7: var_2}
    var_9 = module_0.Resolver(var_0, var_8)
    var_10 = module_1.Load()
    var_11 = {}
    var_12 = 'self'
    var_13 = module_0.Resolver(var_0, var_11, var_12)
    var_14 = module_1.Load()
    var_15 = 'root.T'
    var_16 = "typing.TypeVar('T')"
    var_17 = {var_15: var_16}
    var_18 = module_0.Resolver(var_0, var_17)
    var_19 = 'T'
    var_20 = module_1.Load()
    var_21 = {var_1: var_7, var_7: var_2}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = module_1.Load()



# Parsed testcases at query #17
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
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = {}
    var_29 = module_0.Resolver(var_0, var_28)
    var_30 = 'MyClass'
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True
    var_7 = module_0.Parser()
    var_8 = '_private_name'
    var_9 = {var_8}
    var_10 = 'root._private_name'
    var_11 = var_7.is_public(var_10)
    assert var_11 is False
    var_12 = module_0.Parser()
    var_13 = 'name_in_all'
    var_14 = {var_13}
    var_15 = 'root.name_in_all'
    var_16 = var_12.is_public(var_15)
    assert var_16 is True
    var_17 = module_0.Parser()
    var_18 = 'parent_name'
    var_19 = {var_18}
    var_20 = 'root.parent_name.child'
    var_21 = var_17.is_public(var_20)
    assert var_21 is True
    var_22 = module_0.Parser()
    var_23 = set()
    var_24 = var_22.is_public(var_10)
    assert var_24 is False
    var_25 = module_0.Parser()
    var_26 = {var_18}
    var_27 = 'root.parent_name'
    var_28 = var_25.is_public(var_20)
    assert var_28 is True
    var_29 = module_0.Parser()
    var_30 = 'grandparent_name'
    var_31 = {var_30}
    var_32 = 'root.grandparent_name'
    var_33 = 'root.grandparent_name.parent_name'
    var_34 = 'root.grandparent_name.parent_name.child'
    var_35 = var_29.is_public(var_34)
    assert var_35 is True



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
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
    var_24 = 'Dict'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = {var_9: var_10}
    var_30 = module_0.Resolver(var_0, var_29)
    var_31 = module_1.Load()
    var_32 = module_1.Load()



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'SelfType'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = 'module.submodule'
    var_23 = 'module.submodule.name'
    var_24 = {var_23: var_2}
    var_25 = module_0.Resolver(var_22, var_24)
    var_26 = module_1.Load()



# Parsed testcases at query #21
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
    var_26 = '**kwargs'
    var_27 = 'Any'
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #22
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
    var_9 = 'test.typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Load()
    var_14 = 'str'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'test.typing.Optional'
    var_18 = 'Optional'
    var_19 = {var_17: var_18}
    var_20 = module_0.Resolver(var_0, var_19)
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = {}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = 'Unknown'
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #23
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
    var_16 = module_1.FunctionDef(*var_13)
    var_17 = var_0.api(var_1, var_16)
    var_18 = 'test_async_func'
    var_19 = []
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = module_1.arg()
    var_23 = [var_22]
    var_24 = []
    var_25 = []
    var_26 = []
    var_27 = module_1.arguments(*var_23)
    var_28 = []
    var_29 = []
    var_30 = module_1.AsyncFunctionDef(*var_27)
    var_31 = var_0.api(var_1, var_30)
    var_32 = 'TestClass'
    var_33 = []
    var_34 = []
    var_35 = []
    var_36 = []
    var_37 = module_1.ClassDef()
    var_38 = var_0.api(var_1, var_37)
    var_39 = 'NestedClass'
    var_40 = []
    var_41 = []
    var_42 = []
    var_43 = []
    var_44 = module_1.ClassDef()
    var_45 = var_0.api(var_1, var_37)
    var_46 = 'decorated_func'
    var_47 = []
    var_48 = []
    var_49 = []
    var_50 = []
    var_51 = []
    var_52 = module_1.arguments(*var_48)
    var_53 = []
    var_54 = 'decorator'
    var_55 = module_1.Load()
    var_56 = module_1.Name()
    var_57 = [var_56]
    var_58 = module_1.FunctionDef(*var_52)
    var_59 = var_0.api(var_1, var_58)
    var_60 = 'ClassWithBases'
    var_61 = 'BaseClass'
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = [var_63]
    var_65 = []
    var_66 = []
    var_67 = []
    var_68 = module_1.ClassDef()
    var_69 = var_0.api(var_1, var_68)



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = 'collections'
    var_7 = 'defaultdict'
    var_8 = 0
    var_9 = 'os.path'
    var_10 = 'join'
    var_11 = 'path_join'
    var_12 = 1
    var_13 = 'pkg.subpkg'
    var_14 = 'func'
    var_15 = 2



# Parsed testcases at query #25
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
    var_7 = 'other_module'
    var_8 = module_1.Load()
    var_9 = 'SomeClass'
    var_10 = module_1.Load()
    var_11 = 42
    var_12 = 'value'
    var_13 = module_1.Load()



# Parsed testcases at query #26
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = '\n"""Test module docstring."""\nCONSTANT = 42\ndef function():\n    """Function docstring."""\n    pass\nclass Class:\n    """Class docstring."""\n    pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = True
    var_6 = True
    var_7 = module_0.parse(var_2, var_3)
    var_8 = '\nfrom typing import List\nMyList = List[int]\ndef function() -> MyList:\n    """Function with alias."""\n    pass\n'
    var_9 = module_0.parse(var_2, var_8)
    var_10 = '\nclass TestClass:\n    """Test class."""\n    member: int\n    def method(self) -> None:\n        """Test method."""\n        pass\n'
    var_11 = module_0.parse(var_2, var_10)



# Parsed testcases at query #27
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Union'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = 'str'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'Optional'
    var_20 = module_1.Load()
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = 'NonPEP585'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #28
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    '''Module docstring.'''\n    x = 1\n    y: int = 2\n    def foo(a: int) -> str:\n        '''Function docstring.'''\n        return str(a)\n    class Bar:\n        '''Class docstring.'''\n        z: float = 3.0\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = module_0.Parser()
    var_5 = '\n    from typing import List\n    import os\n    x: List[int] = [1, 2, 3]\n    '
    var_6 = 'test_imports'
    var_7 = var_4.parse(var_6, var_5)
    var_8 = module_0.Parser()
    var_9 = '\n    class MyClass:\n        def method(self, x: int) -> str:\n            return str(x)\n    '
    var_10 = 'test_class'
    var_11 = var_8.parse(var_10, var_9)
    var_12 = module_0.Parser()
    var_13 = '\n    def decorator(func):\n        return func\n    @decorator\n    def decorated_func():\n        pass\n    '
    var_14 = 'test_decorator'
    var_15 = var_12.parse(var_14, var_13)
    var_16 = module_0.Parser()
    var_17 = '\n    from enum import Enum\n    class Color(Enum):\n        RED = 1\n        GREEN = 2\n        BLUE = 3\n    '
    var_18 = 'test_enum'
    var_19 = var_16.parse(var_18, var_17)



# Parsed testcases at query #29
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = False
    var_6 = 'self'
    var_7 = 'TestClass'
    var_8 = module_1.Load()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = True
    var_12 = 'cls'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = '*'
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = '**kwargs'
    var_20 = module_1.Load()
    var_21 = 'z'
    var_22 = module_1.Load()



# Parsed testcases at query #30
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
    var_12 = module_0.Parser()
    var_13 = 'self'
    var_14 = 'Class'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = True
    var_19 = module_0.Parser()
    var_20 = 'cls'
    var_21 = module_1.Load()
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = module_0.Parser()
    var_25 = module_1.Load()
    var_26 = '*'
    var_27 = None
    var_28 = module_1.Load()
    var_29 = '**'
    var_30 = module_1.Load()
    var_31 = module_0.Parser()



# Parsed testcases at query #31
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'test_module.simple_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_0.arguments(*var_5)
    var_10 = None
    var_11 = 'x'
    var_12 = None
    var_13 = 'y'
    var_14 = 'z'
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = 'a'
    var_21 = 'b'
    var_22 = []
    var_23 = []
    var_24 = []
    var_25 = []
    var_26 = []
    var_27 = 2
    var_28 = []
    var_29 = []
    var_30 = []
    var_31 = []
    var_32 = []
    var_33 = 'args'
    var_34 = 'kwargs'
    var_35 = 'int'
    var_36 = module_0.Load()
    var_37 = []
    var_38 = 'self'
    var_39 = []
    var_40 = []
    var_41 = []
    var_42 = True
    var_43 = True



# Parsed testcases at query #32
#--------------------------




# Parsed testcases at query #33
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'SelfType'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'module.A'
    var_23 = 'module.B'
    var_24 = 'final'
    var_25 = {var_22: var_23, var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = 'A'
    var_28 = module_1.Load()



# Parsed testcases at query #34
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'self_ty'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'module.A'
    var_23 = 'module.B'
    var_24 = {var_22: var_23, var_23: var_2}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = 'A'
    var_27 = module_1.Load()



# Parsed testcases at query #35
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = False
    var_6 = 'self'
    var_7 = True
    var_8 = 'cls'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = '*'
    var_15 = '**kwargs'



# Parsed testcases at query #36
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
    var_7 = 'other_module'
    var_8 = module_1.Load()
    var_9 = 'SomeClass'
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = module_1.Load()
    var_13 = 'append'
    var_14 = module_1.Load()
    var_15 = 42
    var_16 = 'some_attr'
    var_17 = module_1.Load()



# Parsed testcases at query #37
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
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = 'typing.Dict'
    var_29 = 'Dict'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_0, var_30)
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()
    var_35 = module_1.Load()
    var_36 = module_1.Load()
    var_37 = {}
    var_38 = module_0.Resolver(var_0, var_37)
    var_39 = module_1.Load()



# Parsed testcases at query #38
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
    var_7 = 'other_module'
    var_8 = module_1.Load()
    var_9 = 'SomeClass'
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Union'
    var_13 = module_1.Load()



# Parsed testcases at query #39
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MyType'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = module_0.Parser()
    var_9 = 'CONST_VALUE'
    var_10 = module_1.Load()
    var_11 = 3.14
    var_12 = None
    var_13 = module_0.Parser()
    var_14 = '__all__'
    var_15 = module_1.Load()
    var_16 = 'public_func'
    var_17 = 'PublicClass'
    var_18 = module_0.Parser()
    var_19 = 'non_const'
    var_20 = module_1.Load()
    var_21 = 'some_value'
    var_22 = module_1.Load()
    var_23 = module_0.Parser()



# Parsed testcases at query #40
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'VAR1'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'VAR2'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = 'VAR3'
    var_14 = module_1.Constant()
    var_15 = 2
    var_16 = module_1.Constant()
    var_17 = [var_14, var_16]
    var_18 = module_1.List()
    var_19 = '__all__'
    var_20 = 'public_func'
    var_21 = module_1.Constant()
    var_22 = 'PublicClass'
    var_23 = module_1.Constant()
    var_24 = [var_21, var_23]
    var_25 = module_1.List()
    var_26 = 'a'
    var_27 = 'b'
    var_28 = module_1.Constant()
    var_29 = 'some_var'
    var_30 = module_1.Load()
    var_31 = module_1.Name()



# Parsed testcases at query #41
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = 'arg1'
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
    var_16 = module_1.FunctionDef(*var_13)
    var_17 = var_0.api(var_1, var_16)
    var_18 = module_0.Parser()
    var_19 = 'TestClass'
    var_20 = []
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = module_1.ClassDef()
    var_25 = var_18.api(var_1, var_24)
    var_26 = module_0.Parser()
    var_27 = 'async_test_func'
    var_28 = []
    var_29 = []
    var_30 = []
    var_31 = []
    var_32 = []
    var_33 = module_1.arguments(*var_29)
    var_34 = []
    var_35 = []
    var_36 = module_1.AsyncFunctionDef(*var_33)
    var_37 = var_26.api(var_1, var_36)
    var_38 = module_0.Parser()
    var_39 = 'decorated_func'
    var_40 = []
    var_41 = []
    var_42 = []
    var_43 = []
    var_44 = []
    var_45 = module_1.arguments(*var_41)
    var_46 = []
    var_47 = 'decorator'
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = [var_49]
    var_51 = module_1.FunctionDef(*var_45)
    var_52 = var_38.api(var_1, var_51)
    var_53 = module_0.Parser()
    var_54 = 'DerivedClass'
    var_55 = 'BaseClass'
    var_56 = module_1.Load()
    var_57 = module_1.Name()
    var_58 = [var_57]
    var_59 = []
    var_60 = []
    var_61 = []
    var_62 = module_1.ClassDef()
    var_63 = var_53.api(var_1, var_62)



# Parsed testcases at query #42
#--------------------------




# Parsed testcases at query #43
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\n"""Test module docstring."""\nx = 1\ny = "hello"\ndef func(a: int) -> str:\n    """Function docstring."""\n    return str(a)\n'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.compile()
    var_5 = True
    var_6 = '\n"""Test module docstring."""\ndef func(a: int) -> str:\n    """Function docstring."""\n    return str(a)\n'
    var_7 = var_0.parse(var_1, var_6)
    var_8 = var_0.compile()
    var_9 = module_0.Parser()
    var_10 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_11 = var_9.parse(var_1, var_10)
    var_12 = var_9.compile()
    var_13 = module_0.Parser()
    var_14 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_15 = var_13.parse(var_1, var_14)
    var_16 = var_13.compile()
    var_17 = module_0.Parser()
    var_18 = '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'
    var_19 = var_17.parse(var_1, var_18)
    var_20 = var_17.compile()
    var_21 = module_0.Parser()
    var_22 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_23 = var_21.parse(var_1, var_22)
    var_24 = var_21.compile()
    var_25 = module_0.Parser()
    var_26 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_27 = var_25.parse(var_1, var_26)
    var_28 = var_25.compile()
    var_29 = module_0.Parser()
    var_30 = '\n"""Test module docstring."""\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_31 = var_29.parse(var_1, var_30)
    var_32 = var_29.compile()
    var_33 = module_0.Parser()
    var_34 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_35 = var_33.parse(var_1, var_34)
    var_36 = var_33.compile()



# Parsed testcases at query #44
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
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = []
    var_33 = []
    var_34 = module_1.Call(*var_32)
    var_35 = module_1.Expr()
    var_36 = [var_35]
    var_37 = module_1.If()
    var_38 = [var_37]
    var_39 = module_0.walk_body(var_38)
    var_40 = list(var_39)
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = [var_42]
    var_44 = module_1.Constant()
    var_45 = module_1.Assign()
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = []
    var_49 = []
    var_50 = module_1.Call(*var_48)
    var_51 = module_1.Expr()
    var_52 = [var_45, var_51]
    var_53 = 'z'
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = [var_55]
    var_57 = 3
    var_58 = module_1.Constant()
    var_59 = module_1.Assign()
    var_60 = [var_59]
    var_61 = []
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = []
    var_65 = []
    var_66 = module_1.Call(*var_64)
    var_67 = module_1.Expr()
    var_68 = [var_67]
    var_69 = 'w'
    var_70 = module_1.Load()
    var_71 = module_1.Name()
    var_72 = [var_71]
    var_73 = 4
    var_74 = module_1.Constant()
    var_75 = module_1.Assign()
    var_76 = [var_75]
    var_77 = module_1.Try()
    var_78 = [var_77]
    var_79 = module_0.walk_body(var_78)
    var_80 = list(var_79)
    var_81 = module_1.Load()
    var_82 = module_1.Name()
    var_83 = [var_82]
    var_84 = module_1.Constant()
    var_85 = module_1.Assign()
    var_86 = module_1.Load()
    var_87 = module_1.Name()
    var_88 = []
    var_89 = []
    var_90 = module_1.Call(*var_88)
    var_91 = module_1.Expr()
    var_92 = module_1.Load()
    var_93 = module_1.Name()
    var_94 = [var_93]
    var_95 = module_1.Constant()
    var_96 = module_1.Assign()
    var_97 = [var_85, var_91, var_96]
    var_98 = True
    var_99 = module_1.Constant()
    var_100 = False
    var_101 = module_1.Constant()
    var_102 = 'a'
    var_103 = module_1.Load()
    var_104 = module_1.Name()
    var_105 = [var_104]
    var_106 = 5
    var_107 = module_1.Constant()
    var_108 = module_1.Assign()
    var_109 = [var_108]
    var_110 = []
    var_111 = module_1.If()
    var_112 = [var_111]
    var_113 = []
    var_114 = module_1.If()
    var_115 = [var_114]
    var_116 = module_0.walk_body(var_115)
    var_117 = list(var_116)
    var_118 = module_1.Constant()
    var_119 = module_1.Load()
    var_120 = module_1.Name()
    var_121 = [var_120]
    var_122 = module_1.Constant()
    var_123 = module_1.Assign()
    var_124 = [var_123]
    var_125 = []
    var_126 = module_1.If()
    var_127 = [var_126]
    var_128 = 'b'
    var_129 = module_1.Load()
    var_130 = module_1.Name()
    var_131 = [var_130]
    var_132 = 6
    var_133 = module_1.Constant()
    var_134 = module_1.Assign()
    var_135 = True
    var_136 = module_1.Constant()
    var_137 = module_1.Load()
    var_138 = module_1.Name()
    var_139 = []
    var_140 = []
    var_141 = module_1.Call(*var_139)
    var_142 = module_1.Expr()
    var_143 = [var_142]
    var_144 = []
    var_145 = module_1.If()
    var_146 = 'c'
    var_147 = module_1.Load()
    var_148 = module_1.Name()
    var_149 = [var_148]
    var_150 = 7
    var_151 = module_1.Constant()
    var_152 = module_1.Assign()
    var_153 = [var_152]
    var_154 = []
    var_155 = []
    var_156 = []
    var_157 = module_1.Try()
    var_158 = [var_134, var_145, var_157]
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



# Parsed testcases at query #45
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
    var_16 = module_1.FunctionDef(*var_13)
    var_17 = var_0.api(var_1, var_16)
    var_18 = module_0.Parser()
    var_19 = 'test_async_func'
    var_20 = []
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = module_1.arg()
    var_24 = [var_23]
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = module_1.arguments(*var_24)
    var_29 = []
    var_30 = []
    var_31 = module_1.AsyncFunctionDef(*var_28)
    var_32 = var_18.api(var_1, var_31)
    var_33 = module_0.Parser()
    var_34 = 'TestClass'
    var_35 = []
    var_36 = []
    var_37 = []
    var_38 = []
    var_39 = module_1.ClassDef()
    var_40 = var_33.api(var_1, var_39)
    var_41 = module_0.Parser()
    var_42 = 'InnerClass'
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = []
    var_47 = module_1.ClassDef()
    var_48 = 'OuterClass'
    var_49 = var_41.api(var_1, var_47, prefix=var_48)
    var_50 = module_0.Parser()
    var_51 = 'test_decorated_func'
    var_52 = []
    var_53 = []
    var_54 = []
    var_55 = []
    var_56 = []
    var_57 = module_1.arguments(*var_53)
    var_58 = []
    var_59 = 'decorator'
    var_60 = module_1.Load()
    var_61 = module_1.Name()
    var_62 = [var_61]
    var_63 = module_1.FunctionDef(*var_57)
    var_64 = var_50.api(var_1, var_63)
    var_65 = module_0.Parser()
    var_66 = 'BaseClass'
    var_67 = module_1.Load()
    var_68 = module_1.Name()
    var_69 = [var_68]
    var_70 = []
    var_71 = []
    var_72 = []
    var_73 = module_1.ClassDef()
    var_74 = var_65.api(var_1, var_73)
    var_75 = module_0.Parser()
    var_76 = 'test_func_with_doc'
    var_77 = []
    var_78 = []
    var_79 = []
    var_80 = []
    var_81 = []
    var_82 = module_1.arguments(*var_78)
    var_83 = 'This is a test function.'
    var_84 = module_1.Constant()
    var_85 = module_1.Expr()
    var_86 = [var_85]
    var_87 = []
    var_88 = module_1.FunctionDef(*var_82)
    var_89 = var_75.api(var_1, var_88)
    var_90 = module_0.Parser()
    var_91 = 'TestClassWithDoc'
    var_92 = []
    var_93 = []
    var_94 = 'This is a test class.'
    var_95 = module_1.Constant()
    var_96 = module_1.Expr()
    var_97 = [var_96]
    var_98 = []
    var_99 = module_1.ClassDef()
    var_100 = var_90.api(var_1, var_99)



# Parsed testcases at query #46
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
    var_6 = 1
    var_7 = None
    var_8 = 'attr2'
    var_9 = 'value'
    var_10 = 'attr3'
    var_11 = []
    var_12 = 'enum.Enum'
    var_13 = module_1.Load()
    var_14 = []
    var_15 = '_private'
    var_16 = '__dunder'
    var_17 = 2
    var_18 = []
    var_19 = 'public'
    var_20 = 'Public'
    var_21 = []



# Parsed testcases at query #47
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test_module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'test_module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'test_module.A'
    var_23 = 'test_module.B'
    var_24 = 'final'
    var_25 = {var_22: var_23, var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25)
    var_27 = 'A'
    var_28 = module_1.Load()



# Parsed testcases at query #48
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'self_ty'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'test_module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'test_module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'final'
    var_23 = {var_1: var_11, var_11: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()



# Parsed testcases at query #49
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
    var_9 = 'listdir'
    var_10 = 1
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 'dd'



# Parsed testcases at query #50
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'TypeAlias'
    var_4 = module_0.Load()
    var_5 = 'int'
    var_6 = module_0.Load()
    var_7 = 42
    var_8 = 'CONSTANT'
    var_9 = module_0.Load()
    var_10 = 3.14
    var_11 = None
    var_12 = '__all__'
    var_13 = module_0.Load()
    var_14 = 'public_func'
    var_15 = 'public_class'
    var_16 = module_0.Load()
    var_17 = 'variable'
    var_18 = module_0.Load()
    var_19 = 100
    var_20 = 'TypedVar'
    var_21 = module_0.Load()
    var_22 = 'hello'
    var_23 = 'str'



# Parsed testcases at query #51
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = {}
    var_8 = 'Self'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = 'module.other'
    var_12 = 'other_replacement'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = 'module.T'
    var_17 = "typing.TypeVar('T')"
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'T'
    var_21 = module_1.Load()
    var_22 = 'final'
    var_23 = {var_1: var_11, var_11: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()



# Parsed testcases at query #52
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.Union'
    var_2 = 'test.Optional'
    var_3 = 'typing.Union'
    var_4 = 'typing.Optional'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'Union'
    var_8 = module_1.Load()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = 'Optional'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'test.List'
    var_20 = 'typing.List'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = 'List'
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = 'test.Custom'
    var_28 = 'typing.Custom'
    var_29 = {var_27: var_28}
    var_30 = module_0.Resolver(var_0, var_29)
    var_31 = 'Custom'
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()



# Parsed testcases at query #53
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
    var_11 = var_0.func_api(var_1, var_2, var_8, var_9, has_self=var_10, cls_method=var_10)
    var_12 = '|||\n|-|-|-|\n|return|Any|\n'
    var_13 = var_0.doc[var_2]
    var_14 = module_0.Parser()
    var_15 = 'test_module'
    var_16 = 'test_module.complex_func'
    var_17 = 'a'
    var_18 = None
    var_19 = 'b'
    var_20 = 'c'
    var_21 = 'd'
    var_22 = 1
    var_23 = 2
    var_24 = 'str'
    var_25 = var_14.func_api(var_15, var_16, var_8, var_9, has_self=var_10, cls_method=var_10)
    var_26 = '|||||\n|-|-|-|-|-|\n|a|b|/|c|*|\n|d|return|\n|Any|Any|Any|Any|Any|\n|1|str|\n'
    var_27 = var_14.doc[var_16]
    var_28 = module_0.Parser()
    var_29 = 'test_module'
    var_30 = 'test_module.MyClass.method'
    var_31 = []
    var_32 = 'self'
    var_33 = 'x'
    var_34 = []
    var_35 = []
    var_36 = []
    var_37 = None
    var_38 = True
    var_39 = var_28.func_api(var_29, var_30, var_8, var_37, has_self=var_38, cls_method=var_10)
    var_40 = '|||\n|-|-|-|\n|self|x|return|\n|Self|Any|Any|\n'
    var_41 = var_28.doc[var_30]
    var_42 = module_0.Parser()
    var_43 = 'test_module'
    var_44 = 'test_module.MyClass.class_method'
    var_45 = []
    var_46 = 'cls'
    var_47 = []
    var_48 = []
    var_49 = []
    var_50 = None
    var_51 = True
    var_52 = True
    var_53 = var_42.func_api(var_43, var_44, var_8, var_50, has_self=var_51, cls_method=var_52)
    var_54 = '|||\n|-|-|-|\n|cls|x|return|\n|type[Self]|Any|Any|\n'
    var_55 = var_42.doc[var_44]
    var_56 = module_0.Parser()
    var_57 = 'test_module'
    var_58 = 'test_module.var_args_func'
    var_59 = []
    var_60 = []
    var_61 = []
    var_62 = []
    var_63 = 'args'
    var_64 = 'kwargs'
    var_65 = None
    var_66 = var_56.func_api(var_57, var_58, var_8, var_65, has_self=var_10, cls_method=var_10)
    var_67 = '||||\n|-|-|-|-|\n|x|*args|**kwargs|return|\n|Any|Any|Any|Any|\n'
    var_68 = var_56.doc[var_58]



# Parsed testcases at query #54
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
    var_10 = 2
    var_11 = module_1.Constant()
    var_12 = module_1.Expr()
    var_13 = [var_9, var_12]
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = 'y'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = [var_20]
    var_22 = 3
    var_23 = module_1.Constant()
    var_24 = module_1.Assign()
    var_25 = [var_24]
    var_26 = 4
    var_27 = module_1.Constant()
    var_28 = module_1.Expr()
    var_29 = [var_28]
    var_30 = module_1.If()
    var_31 = [var_30]
    var_32 = module_0.walk_body(var_31)
    var_33 = list(var_32)
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = [var_35]
    var_37 = module_1.Constant()
    var_38 = module_1.Assign()
    var_39 = module_1.Constant()
    var_40 = module_1.Expr()
    var_41 = [var_38, var_40]
    var_42 = 'z'
    var_43 = module_1.Load()
    var_44 = module_1.Name()
    var_45 = [var_44]
    var_46 = 5
    var_47 = module_1.Constant()
    var_48 = module_1.Assign()
    var_49 = [var_48]
    var_50 = 'type'
    var_51 = 'name'
    var_52 = 'body'
    var_53 = 'Exception'
    var_54 = module_1.Load()
    var_55 = module_1.Name()
    var_56 = 'e'
    var_57 = 6
    var_58 = module_1.Constant()
    var_59 = module_1.Expr()
    var_60 = [var_59]
    var_61 = {var_50: var_55, var_51: var_56, var_52: var_60}
    var_62 = [var_61]
    var_63 = 7
    var_64 = module_1.Constant()
    var_65 = module_1.Expr()
    var_66 = [var_65]
    var_67 = 8
    var_68 = module_1.Constant()
    var_69 = module_1.Expr()
    var_70 = [var_69]
    var_71 = module_1.Try()
    var_72 = [var_71]
    var_73 = module_0.walk_body(var_72)
    var_74 = list(var_73)
    var_75 = module_1.Load()
    var_76 = module_1.Name()
    var_77 = [var_76]
    var_78 = module_1.Constant()
    var_79 = module_1.Assign()
    var_80 = module_1.Constant()
    var_81 = module_1.Expr()
    var_82 = module_1.Constant()
    var_83 = module_1.Expr()
    var_84 = module_1.Constant()
    var_85 = module_1.Expr()
    var_86 = [var_79, var_81, var_83, var_85]
    var_87 = True
    var_88 = module_1.Constant()
    var_89 = 'a'
    var_90 = module_1.Load()
    var_91 = module_1.Name()
    var_92 = [var_91]
    var_93 = 9
    var_94 = module_1.Constant()
    var_95 = module_1.Assign()
    var_96 = [var_95]
    var_97 = []
    var_98 = []
    var_99 = []
    var_100 = module_1.Try()
    var_101 = 10
    var_102 = module_1.Constant()
    var_103 = module_1.Expr()
    var_104 = [var_100, var_103]
    var_105 = False
    var_106 = module_1.Constant()
    var_107 = 11
    var_108 = module_1.Constant()
    var_109 = module_1.Expr()
    var_110 = [var_109]
    var_111 = []
    var_112 = module_1.If()
    var_113 = [var_112]
    var_114 = module_1.If()
    var_115 = [var_114]
    var_116 = module_0.walk_body(var_115)
    var_117 = list(var_116)
    var_118 = module_1.Load()
    var_119 = module_1.Name()
    var_120 = [var_119]
    var_121 = module_1.Constant()
    var_122 = module_1.Assign()
    var_123 = module_1.Constant()
    var_124 = module_1.Expr()
    var_125 = module_1.Constant()
    var_126 = module_1.Expr()
    var_127 = [var_122, var_124, var_126]



# Parsed testcases at query #55
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST_VAR'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = 'test_module.CONST_VAR'
    var_9 = 'ANOTHER_CONST'
    var_10 = 'hello'
    var_11 = 'str'
    var_12 = 'test_module.ANOTHER_CONST'
    var_13 = 'INFERRED_CONST'
    var_14 = 2
    var_15 = 3
    var_16 = 'test_module.INFERRED_CONST'
    var_17 = '__all__'
    var_18 = 'public_func'
    var_19 = 'PublicClass'
    var_20 = 'non_const'
    var_21 = 'test_module.non_const'
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 'test_module.a'
    var_25 = 'test_module.b'



# Parsed testcases at query #56
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = False
    var_6 = 'self'
    var_7 = True
    var_8 = 'cls'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = 'TestClass'
    var_14 = module_1.Load()
    var_15 = 'Self'
    var_16 = module_1.Load()
    var_17 = '*'
    var_18 = '**kwargs'



# Parsed testcases at query #57
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
    var_9 = module_1.Load()
    var_10 = 'typing.Union'
    var_11 = 'Union'
    var_12 = {var_10: var_11}
    var_13 = module_0.Resolver(var_0, var_12)
    var_14 = module_1.Load()
    var_15 = 'str'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'typing.Optional'
    var_19 = 'Optional'
    var_20 = {var_18: var_19}
    var_21 = module_0.Resolver(var_0, var_20)
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = 'typing.Dict'
    var_25 = 'Dict'
    var_26 = {var_24: var_25}
    var_27 = module_0.Resolver(var_0, var_26)
    var_28 = module_1.Load()
    var_29 = module_1.Load()
    var_30 = module_1.Load()
    var_31 = {var_1: var_2}
    var_32 = module_0.Resolver(var_0, var_31)
    var_33 = module_1.Load()
    var_34 = module_1.Load()
    var_35 = module_1.Load()



# Parsed testcases at query #58
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = 'public_attr'
    var_6 = 42
    var_7 = None
    var_8 = '_private_attr'
    var_9 = 'private'
    var_10 = 'class_attr'
    var_11 = 3.14
    var_12 = module_0.Parser()
    var_13 = 'test_module'
    var_14 = 'test_module.TestEnum'
    var_15 = 'enum.Enum'
    var_16 = module_1.Load()
    var_17 = 'VALUE1'
    var_18 = 1
    var_19 = 'VALUE2'
    var_20 = 2
    var_21 = '_private'
    var_22 = module_0.Parser()
    var_23 = 'test_module'
    var_24 = 'test_module.TestClass'
    var_25 = []
    var_26 = 'attr1'
    var_27 = 'attr2'
    var_28 = module_0.Parser()
    var_29 = 'test_module'
    var_30 = 'test_module.EmptyClass'
    var_31 = []
    var_32 = []
    var_33 = var_28.class_api(var_29, var_30, var_31, var_32)



# Parsed testcases at query #59
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = False
    var_6 = 'self'
    var_7 = True
    var_8 = 'TestClass'
    var_9 = module_1.Load()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = 'cls'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = '*'
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = '**'
    var_20 = 'z'
    var_21 = 'float'
    var_22 = module_1.Load()
    var_23 = 'return'
    var_24 = 'bool'
    var_25 = module_1.Load()
    var_26 = 'a'
    var_27 = module_1.Load()
    var_28 = 'b'
    var_29 = 'c'
    var_30 = module_1.Load()



# Parsed testcases at query #60
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
    var_9 = 'attr2'
    var_10 = 'str'
    var_11 = module_1.Load()
    var_12 = 'attr3'
    var_13 = 42
    var_14 = 'Any'
    var_15 = 0



# Parsed testcases at query #61
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'typing'
    var_1 = module_0.Load()
    var_2 = 'List'
    var_3 = module_0.Load()
    var_4 = 'test_module'
    var_5 = 'test_module.typing'
    var_6 = {var_5: var_0}
    var_7 = module_1.Resolver(var_4, var_6)
    var_8 = 'other_module'
    var_9 = module_0.Load()
    var_10 = 'SomeClass'
    var_11 = module_0.Load()
    var_12 = {}
    var_13 = module_1.Resolver(var_4, var_12)
    var_14 = 42
    var_15 = 'attr'
    var_16 = module_0.Load()
    var_17 = {}
    var_18 = module_1.Resolver(var_4, var_17)



# Parsed testcases at query #62
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'root'
    var_5 = False
    var_6 = module_0.Parser()
    var_7 = 'self'
    var_8 = True
    var_9 = module_0.Parser()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = 'str'
    var_13 = module_1.Load()
    var_14 = module_0.Parser()
    var_15 = 'cls'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_0.Parser()
    var_19 = module_1.Load()
    var_20 = '*'
    var_21 = module_1.Load()
    var_22 = '**'
    var_23 = 'return'
    var_24 = 'bool'
    var_25 = module_1.Load()
    var_26 = module_0.Parser()
    var_27 = 'Self'
    var_28 = module_1.Load()
    var_29 = 'CustomType'



# Parsed testcases at query #63
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.name'
    var_2 = 'replacement'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = module_1.Load()
    var_8 = {}
    var_9 = 'self_ty'
    var_10 = module_0.Resolver(var_0, var_8, var_9)
    var_11 = module_1.Load()
    var_12 = 'test_module.other'
    var_13 = 'other_replacement'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14, var_4)
    var_16 = module_1.Load()
    var_17 = {var_1: var_1}
    var_18 = module_0.Resolver(var_0, var_17, var_4)
    var_19 = module_1.Load()
    var_20 = 'test_module.T'
    var_21 = "typing.TypeVar('T')"
    var_22 = {var_20: var_21}
    var_23 = module_0.Resolver(var_0, var_22, var_4)
    var_24 = 'T'
    var_25 = module_1.Load()



# Parsed testcases at query #64
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'sys'
    var_3 = None
    var_4 = 'numpy'
    var_5 = 'np'
    var_6 = 'collections'
    var_7 = 'defaultdict'
    var_8 = 0
    var_9 = 'os.path'
    var_10 = 'join'
    var_11 = 'path_join'
    var_12 = 'sibling'
    var_13 = 'helper'
    var_14 = 1



# Parsed testcases at query #65
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'
    var_3 = 1
    var_4 = module_0.Constant()
    var_5 = 2
    var_6 = module_0.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_0.Tuple()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'tuple[int, int]'
    var_10 = module_0.Constant()
    var_11 = module_0.Constant()
    var_12 = [var_10, var_11]
    var_13 = module_0.List()
    var_14 = module_1.const_type(var_13)
    assert var_14 == 'list[int, int]'
    var_15 = module_0.Constant()
    var_16 = module_0.Constant()
    var_17 = [var_15, var_16]
    var_18 = module_0.Set()
    var_19 = module_1.const_type(var_18)
    assert var_19 == 'set[int, int]'
    var_20 = module_0.Constant()
    var_21 = module_0.Constant()
    var_22 = [var_20, var_21]
    var_23 = 'a'
    var_24 = module_0.Constant()
    var_25 = 'b'
    var_26 = module_0.Constant()
    var_27 = [var_24, var_26]
    var_28 = module_0.Dict()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'dict[int, str]'
    var_30 = 'int'
    var_31 = module_0.Name()
    var_32 = module_0.Call()
    var_33 = module_1.const_type(var_32)
    assert var_33 == 'int'
    var_34 = 'module'
    var_35 = module_0.Name()
    var_36 = module_0.Attribute()
    var_37 = module_0.Call()
    var_38 = module_1.const_type(var_37)
    assert var_38 == 'int'
    var_39 = module_0.Constant()
    var_40 = module_0.BitOr()
    var_41 = module_0.Constant()
    var_42 = module_0.BinOp()
    var_43 = module_1.const_type(var_42)
    assert var_43 == 'Any'



# Parsed testcases at query #66
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = 'return'
    var_5 = 'str'
    var_6 = module_1.Load()
    var_7 = 'module'
    var_8 = False
    var_9 = module_0.Parser()
    var_10 = 'self'
    var_11 = 'Class'
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = True
    var_16 = module_0.Parser()
    var_17 = 'cls'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = module_0.Parser()
    var_22 = module_1.Load()
    var_23 = '*'
    var_24 = None
    var_25 = 'y'
    var_26 = module_1.Load()
    var_27 = '**'
    var_28 = 'bool'
    var_29 = module_1.Load()
    var_30 = module_0.Parser()



# Parsed testcases at query #67
#--------------------------




# Parsed testcases at query #68
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
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = True
    var_14 = module_0.Constant()
    var_15 = 'y'
    var_16 = module_0.Load()
    var_17 = module_0.Name()
    var_18 = [var_17]
    var_19 = 2
    var_20 = module_0.Constant()
    var_21 = module_0.Assign()
    var_22 = [var_21]
    var_23 = []
    var_24 = module_0.If()
    var_25 = [var_24]
    var_26 = module_1.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_27[var_11]
    var_30 = 'z'
    var_31 = module_0.Load()
    var_32 = module_0.Name()
    var_33 = [var_32]
    var_34 = 3
    var_35 = module_0.Constant()
    var_36 = module_0.Assign()
    var_37 = [var_36]
    var_38 = []
    var_39 = []
    var_40 = []
    var_41 = module_0.Try()
    var_42 = [var_41]
    var_43 = module_1.walk_body(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = var_44[var_11]
    var_47 = True
    var_48 = module_0.Constant()
    var_49 = False
    var_50 = module_0.Constant()
    var_51 = 'a'
    var_52 = module_0.Load()
    var_53 = module_0.Name()
    var_54 = [var_53]
    var_55 = 4
    var_56 = module_0.Constant()
    var_57 = module_0.Assign()
    var_58 = [var_57]
    var_59 = []
    var_60 = module_0.If()
    var_61 = [var_60]
    var_62 = []
    var_63 = module_0.If()
    var_64 = [var_63]
    var_65 = module_1.walk_body(var_64)
    var_66 = list(var_65)
    var_67 = len(var_66)
    assert var_67 == 1
    var_68 = var_66[var_49]
    var_69 = 'b'
    var_70 = module_0.Load()
    var_71 = module_0.Name()
    var_72 = [var_71]
    var_73 = 5
    var_74 = module_0.Constant()
    var_75 = module_0.Assign()
    var_76 = [var_75]
    var_77 = 'type'
    var_78 = 'body'
    var_79 = 'Exception'
    var_80 = module_0.Load()
    var_81 = module_0.Name()
    var_82 = 'c'
    var_83 = module_0.Load()
    var_84 = module_0.Name()
    var_85 = [var_84]
    var_86 = 6
    var_87 = module_0.Constant()
    var_88 = module_0.Assign()
    var_89 = [var_88]
    var_90 = {var_77: var_81, var_78: var_89}
    var_91 = [var_90]
    var_92 = []
    var_93 = []
    var_94 = module_0.Try()
    var_95 = [var_94]
    var_96 = module_1.walk_body(var_95)
    var_97 = list(var_96)
    var_98 = len(var_97)
    assert var_98 == 2
    var_99 = []
    var_100 = module_1.walk_body(var_99)
    var_101 = list(var_100)
    var_102 = len(var_101)
    assert var_102 == 0



# Parsed testcases at query #69
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
    var_30 = 'SomeType'
    var_31 = module_1.Load()
    var_32 = module_1.Load()
    var_33 = module_1.Load()



# Parsed testcases at query #70
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
    var_12 = module_0.Constant()
    var_13 = 2
    var_14 = module_0.Constant()
    var_15 = [var_12, var_14]
    var_16 = module_0.Tuple()
    var_17 = module_1.const_type(var_16)
    assert var_17 == 'tuple[int, int]'
    var_18 = module_0.Constant()
    var_19 = 'a'
    var_20 = module_0.Constant()
    var_21 = [var_18, var_20]
    var_22 = module_0.Tuple()
    var_23 = module_1.const_type(var_22)
    assert var_23 == 'tuple[Any, Any]'
    var_24 = module_0.Constant()
    var_25 = module_0.Constant()
    var_26 = [var_24, var_25]
    var_27 = module_0.List()
    var_28 = module_1.const_type(var_27)
    assert var_28 == 'list[int, int]'
    var_29 = module_0.Constant()
    var_30 = module_0.Constant()
    var_31 = [var_29, var_30]
    var_32 = module_0.Set()
    var_33 = module_1.const_type(var_32)
    assert var_33 == 'set[int, int]'
    var_34 = module_0.Constant()
    var_35 = [var_34]
    var_36 = module_0.Constant()
    var_37 = [var_36]
    var_38 = module_0.Dict()
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'dict[str, int]'
    var_40 = module_0.Constant()
    var_41 = module_0.Constant()
    var_42 = [var_40, var_41]
    var_43 = module_0.Constant()
    var_44 = 'b'
    var_45 = module_0.Constant()
    var_46 = [var_43, var_45]
    var_47 = module_0.Dict()
    var_48 = module_1.const_type(var_47)
    assert var_48 == 'dict[Any, Any]'
    var_49 = 'int'
    var_50 = module_0.Name()
    var_51 = module_0.Call()
    var_52 = module_1.const_type(var_51)
    assert var_52 == 'int'
    var_53 = 'x'
    var_54 = module_0.Name()
    var_55 = module_0.Attribute()
    var_56 = module_0.Call()
    var_57 = module_1.const_type(var_56)
    assert var_57 == 'int'
    var_58 = module_0.Name()
    var_59 = module_1.const_type(var_58)
    assert var_59 == 'Any'
    var_60 = module_0.BinOp()
    var_61 = module_1.const_type(var_60)
    assert var_61 == 'Any'



# Parsed testcases at query #71
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'typing.Union'
    var_2 = 'Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = module_1.Load()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'typing.Optional'
    var_13 = 'Optional'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14)
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = 'typing.List'
    var_20 = 'List'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = module_1.Load()
    var_24 = module_1.Load()
    var_25 = module_1.Load()
    var_26 = {}
    var_27 = module_0.Resolver(var_0, var_26)
    var_28 = module_1.Load()
    var_29 = module_1.Load()
    var_30 = module_1.Load()



# Parsed testcases at query #72
#--------------------------




# Parsed testcases at query #73
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
    var_7 = 1
    var_8 = module_1.Load()
    var_9 = 'typing.Union'
    var_10 = 'Union'
    var_11 = {var_9: var_10}
    var_12 = module_0.Resolver(var_0, var_11)
    var_13 = module_1.Load()
    var_14 = 2
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
    var_27 = 'SomeType'
    var_28 = module_1.Load()
    var_29 = module_1.Load()



# Parsed testcases at query #74
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



# Parsed testcases at query #75
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func1'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = []
    var_11 = module_1.FunctionDef(*var_8)
    var_12 = 'test_module.test_func1'
    var_13 = var_11.args
    var_14 = var_11.returns
    var_15 = False
    var_16 = var_0.func_api(var_1, var_12, var_13, var_14, has_self=var_15, cls_method=var_15)
    var_17 = var_0.doc[var_12]
    var_18 = 'test_func2'
    var_19 = 'arg1'
    var_20 = 'int'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = module_1.arg()
    var_24 = [var_23]
    var_25 = 'arg2'
    var_26 = 'str'
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = module_1.arg()
    var_30 = [var_29]
    var_31 = []
    var_32 = []
    var_33 = []
    var_34 = module_1.arguments(*var_30)
    var_35 = 'bool'
    var_36 = module_1.Load()
    var_37 = module_1.Name()
    var_38 = []
    var_39 = module_1.FunctionDef(*var_34)
    var_40 = 'test_module.test_func2'
    var_41 = var_39.args
    var_42 = var_39.returns
    var_43 = var_0.func_api(var_1, var_40, var_41, var_42, has_self=var_15, cls_method=var_15)
    var_44 = var_0.doc[var_40]
    var_45 = 'test_func3'
    var_46 = []
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = module_1.arg()
    var_50 = [var_49]
    var_51 = []
    var_52 = []
    var_53 = 10
    var_54 = module_1.Constant()
    var_55 = [var_54]
    var_56 = module_1.arguments(*var_50)
    var_57 = []
    var_58 = module_1.FunctionDef(*var_56)
    var_59 = 'test_module.test_func3'
    var_60 = var_58.args
    var_61 = var_58.returns
    var_62 = var_0.func_api(var_1, var_59, var_60, var_61, has_self=var_15, cls_method=var_15)
    var_63 = var_0.doc[var_59]
    var_64 = 'test_func4'
    var_65 = []
    var_66 = []
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = 'args'
    var_71 = module_1.arg()
    var_72 = 'kwargs'
    var_73 = module_1.arg()
    var_74 = module_1.arguments(*var_66)
    var_75 = []
    var_76 = module_1.FunctionDef(*var_74)
    var_77 = 'test_module.test_func4'
    var_78 = var_76.args
    var_79 = var_76.returns
    var_80 = var_0.func_api(var_1, var_77, var_78, var_79, has_self=var_15, cls_method=var_15)
    var_81 = var_0.doc[var_77]
    var_82 = 'test_func5'
    var_83 = []
    var_84 = 'self'
    var_85 = module_1.arg()
    var_86 = [var_85]
    var_87 = []
    var_88 = []
    var_89 = []
    var_90 = module_1.arguments(*var_86)
    var_91 = 'classmethod'
    var_92 = module_1.Load()
    var_93 = module_1.Name()
    var_94 = [var_93]
    var_95 = module_1.FunctionDef(*var_90)
    var_96 = 'test_module.test_func5'
    var_97 = var_95.args
    var_98 = var_95.returns
    var_99 = True
    var_100 = var_0.func_api(var_1, var_96, var_97, var_98, has_self=var_99, cls_method=var_99)
    var_101 = var_0.doc[var_96]



# Parsed testcases at query #76
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'typing.List'
    var_2 = 'list'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'typing'
    var_6 = module_1.Load()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = 'other_module'
    var_10 = module_1.Load()
    var_11 = 'SomeClass'
    var_12 = module_1.Load()
    var_13 = 'module'
    var_14 = module_1.Load()
    var_15 = 'submodule'
    var_16 = module_1.Load()
    var_17 = 'Class'
    var_18 = module_1.Load()



# Parsed testcases at query #77
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'
    var_3 = 1
    var_4 = module_0.Constant()
    var_5 = 2
    var_6 = module_0.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_0.Tuple()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'tuple[int, int]'
    var_10 = 'a'
    var_11 = module_0.Constant()
    var_12 = 'b'
    var_13 = module_0.Constant()
    var_14 = [var_11, var_13]
    var_15 = module_0.List()
    var_16 = module_1.const_type(var_15)
    assert var_16 == 'list[str, str]'
    var_17 = module_0.Constant()
    var_18 = module_0.Constant()
    var_19 = [var_17, var_18]
    var_20 = module_0.Set()
    var_21 = module_1.const_type(var_20)
    assert var_21 == 'set[float, float]'
    var_22 = 'x'
    var_23 = module_0.Constant()
    var_24 = 'y'
    var_25 = module_0.Constant()
    var_26 = [var_23, var_25]
    var_27 = module_0.Constant()
    var_28 = module_0.Constant()
    var_29 = [var_27, var_28]
    var_30 = module_0.Dict()
    var_31 = module_1.const_type(var_30)
    assert var_31 == 'dict[str, str, int, int]'
    var_32 = 'int'
    var_33 = module_0.Name()
    var_34 = module_0.Call()
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'int'
    var_36 = 'typing'
    var_37 = module_0.Name()
    var_38 = 'List'
    var_39 = module_0.Attribute()
    var_40 = module_0.Call()
    var_41 = module_1.const_type(var_40)
    assert var_41 == 'typing.List'
    var_42 = module_0.Constant()
    var_43 = module_0.BitOr()
    var_44 = module_0.Constant()
    var_45 = module_0.BinOp()
    var_46 = module_1.const_type(var_45)
    assert var_46 == 'Any'



# Parsed testcases at query #78
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = 'os.path'
    var_7 = 'join'
    var_8 = 0
    var_9 = 'path'
    var_10 = 1
    var_11 = 'test'
    var_12 = 'path.join'
    var_13 = 'ospath'



# Parsed testcases at query #79
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
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'typing.Union'
    var_12 = 'Union'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_0, var_13)
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'typing.Optional'
    var_22 = 'Optional'
    var_23 = {var_21: var_22}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = module_1.Load()
    var_26 = module_1.Load()
    var_27 = module_1.Load()
    var_28 = 'typing.Dict'
    var_29 = 'Dict'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_0, var_30)
    var_32 = module_1.Load()
    var_33 = module_1.Load()
    var_34 = module_1.Load()
    var_35 = module_1.Load()
    var_36 = module_1.Load()



# Parsed testcases at query #80
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()
    var_7 = 'test.other'
    var_8 = 'str'
    var_9 = {var_7: var_8}
    var_10 = module_0.Resolver(var_0, var_9)
    var_11 = module_1.Load()
    var_12 = {}
    var_13 = 'self_ty'
    var_14 = module_0.Resolver(var_0, var_12, var_13)
    var_15 = module_1.Load()
    var_16 = 'test.TypeVar'
    var_17 = 'typing.TypeVar'
    var_18 = {var_16: var_17}
    var_19 = module_0.Resolver(var_0, var_18)
    var_20 = 'TypeVar'
    var_21 = module_1.Load()
    var_22 = 'test.nested.name'
    var_23 = 'float'
    var_24 = {var_22: var_23}
    var_25 = module_0.Resolver(var_0, var_24)
    var_26 = 'nested.name'
    var_27 = module_1.Load()



# Parsed testcases at query #81
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
    var_6 = 1
    var_7 = None
    var_8 = 'attr2'
    var_9 = 'hello'
    var_10 = 'attr3'
    var_11 = 3.14
    var_12 = 'Any'
    var_13 = 0
    var_14 = ''
    var_15 = set()
    var_16 = 'Bases\n---\n| BaseClass |\n\nMembers\n-------\n| Name | Type |\n|------|------|\n| `attr1` | `attr1` |\n| `attr3` | `Any` |\n'



# Parsed testcases at query #82
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
    var_23 = {}
    var_24 = module_0.Resolver(var_0, var_23)
    var_25 = 'SomeType'
    var_26 = module_1.Load()
    var_27 = module_1.Load()



# Parsed testcases at query #83
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = 'a'
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = module_1.arg()
    var_8 = [var_7]
    var_9 = 'c'
    var_10 = module_1.arg()
    var_11 = [var_10]
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = [var_13]
    var_15 = 2
    var_16 = module_1.Constant()
    var_17 = [var_16]
    var_18 = module_1.arguments(*var_8)
    var_19 = []
    var_20 = module_0.Parser()
    var_21 = 'async_test_function'
    var_22 = module_1.arg()
    var_23 = [var_22]
    var_24 = module_1.Constant()
    var_25 = [var_24]
    var_26 = module_1.arguments(*var_23)
    var_27 = []
    var_28 = module_0.Parser()
    var_29 = 'TestClass'
    var_30 = []
    var_31 = []
    var_32 = module_0.Parser()
    var_33 = 'InnerClass'
    var_34 = []
    var_35 = []
    var_36 = 'OuterClass'
    var_37 = module_0.Parser()
    var_38 = 'decorated_function'
    var_39 = module_1.arguments()
    var_40 = 'decorator'
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = [var_42]
    var_44 = module_0.Parser()
    var_45 = 'DerivedClass'
    var_46 = 'BaseClass'
    var_47 = module_1.Load()
    var_48 = module_1.Name()
    var_49 = [var_48]
    var_50 = []
    var_51 = module_0.Parser()
    var_52 = 'ClassWithMembers'
    var_53 = []
    var_54 = 'member'
    var_55 = 'int'
    var_56 = module_1.Load()
    var_57 = module_1.Name()
    var_58 = None
    var_59 = []
    var_60 = module_0.Parser()
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = []



# Parsed testcases at query #84
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
    var_13 = 'Any'
    var_14 = ''
    var_15 = 0
    var_16 = set()
    var_17 = 'Bases\n---\n`BaseClass`\n\nMembers\n-------\nName | Type\n--- | ---\n`attr1` | `int`\n`attr2` | `int`\n'



# Parsed testcases at query #85
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
    var_12 = module_0.Constant()
    var_13 = 2
    var_14 = module_0.Constant()
    var_15 = [var_12, var_14]
    var_16 = module_0.Load()
    var_17 = module_0.Tuple()
    var_18 = module_1.const_type(var_17)
    assert var_18 == 'tuple[int, int]'
    var_19 = module_0.Constant()
    var_20 = module_0.Constant()
    var_21 = [var_19, var_20]
    var_22 = module_0.Load()
    var_23 = module_0.List()
    var_24 = module_1.const_type(var_23)
    assert var_24 == 'list[int, int]'
    var_25 = module_0.Constant()
    var_26 = module_0.Constant()
    var_27 = [var_25, var_26]
    var_28 = module_0.Set()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'set[int, int]'
    var_30 = 'a'
    var_31 = module_0.Constant()
    var_32 = 'b'
    var_33 = module_0.Constant()
    var_34 = [var_31, var_33]
    var_35 = module_0.Constant()
    var_36 = module_0.Constant()
    var_37 = [var_35, var_36]
    var_38 = module_0.Dict()
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'dict[str, int]'
    var_40 = 'int'
    var_41 = module_0.Load()
    var_42 = module_0.Name()
    var_43 = '42'
    var_44 = module_0.Constant()
    var_45 = [var_44]
    var_46 = []
    var_47 = module_0.Call(*var_45)
    var_48 = module_1.const_type(var_47)
    assert var_48 == 'int'
    var_49 = 'builtins'
    var_50 = module_0.Load()
    var_51 = module_0.Name()
    var_52 = 'str'
    var_53 = module_0.Load()
    var_54 = module_0.Attribute()
    var_55 = module_0.Constant()
    var_56 = [var_55]
    var_57 = []
    var_58 = module_0.Call(*var_56)
    var_59 = module_1.const_type(var_58)
    assert var_59 == 'str'
    var_60 = module_0.Constant()
    var_61 = module_0.BitOr()
    var_62 = module_0.Constant()
    var_63 = module_0.BinOp()
    var_64 = module_1.const_type(var_63)
    assert var_64 == 'Any'
    var_65 = None
    var_66 = module_1.const_type(var_65)
    assert var_66 == 'Any'



# Parsed testcases at query #86
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MyType'
    var_3 = module_1.Load()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = 42
    var_7 = 1
    var_8 = 'CONST'
    var_9 = module_1.Load()
    var_10 = 3.14
    var_11 = None
    var_12 = '__all__'
    var_13 = module_1.Load()
    var_14 = 'public_func'
    var_15 = 'public_var'
    var_16 = module_1.Load()
    var_17 = 'var'
    var_18 = module_1.Load()
    var_19 = 'value'
    var_20 = module_1.Load()
    var_21 = 'list'
    var_22 = module_1.Load()
    var_23 = 0
    var_24 = module_1.Load()



