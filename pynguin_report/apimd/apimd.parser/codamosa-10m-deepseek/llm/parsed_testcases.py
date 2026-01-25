####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.ExampleClass'
    var_3 = 'object'
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = 42
    var_7 = True
    var_8 = 'attr2'
    var_9 = 'str'
    var_10 = 'attr3'
    var_11 = 'float'
    var_12 = 3.14
    var_13 = var_0.doc[var_2]
    var_14 = '### class ExampleClass\n\n'
    var_15 = 'enum.Enum'
    var_16 = 'ENUM_VALUE1'
    var_17 = 'ENUM_VALUE2'
    var_18 = 2
    var_19 = 'test_module.ExampleEnum'



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'example'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 'example_name'
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = 'invalid name with spaces'
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'example_name + 1'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello, World!')"
    var_1 = "```python\n>>> print('Hello, World!')\n```"
    var_2 = module_0.doctest(var_0)
    var_3 = '>>> x = 5\n>>> y = 10\n>>> print(x + y)'
    var_4 = '```python\n>>> x = 5\n>>> y = 10\n>>> print(x + y)\n```'
    var_5 = module_0.doctest(var_3)
    var_6 = "This is not a doctest.\n>>> print('This is a doctest.')"
    var_7 = "This is not a doctest.\n```python\n>>> print('This is a doctest.')\n```"
    var_8 = module_0.doctest(var_6)
    var_9 = ">>> print('End of doctest')"
    var_10 = "```python\n>>> print('End of doctest')\n```"
    var_11 = module_0.doctest(var_9)
    var_12 = ''
    var_13 = ''
    var_14 = module_0.doctest(var_12)
    var_15 = 'This is just text.\nMore text.'
    var_16 = 'This is just text.\nMore text.'
    var_17 = module_0.doctest(var_15)
    var_18 = "Text before doctest.\n>>> print('Doctest line 1')\nText between.\n>>> print('Doctest line 2')"
    var_19 = "Text before doctest.\n```python\n>>> print('Doctest line 1')\n```\nText between.\n```python\n>>> print('Doctest line 2')\n```"
    var_20 = module_0.doctest(var_18)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'ANOTHER_CONST'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'a'
    var_13 = 'b'
    var_14 = module_1.Constant()
    var_15 = 2
    var_16 = module_1.Constant()
    var_17 = [var_14, var_16]
    var_18 = module_1.Load()
    var_19 = module_1.Tuple()
    var_20 = '__all__'
    var_21 = 'func1'
    var_22 = module_1.Constant()
    var_23 = 'func2'
    var_24 = module_1.Constant()
    var_25 = [var_22, var_24]
    var_26 = module_1.Load()
    var_27 = module_1.List()
    var_28 = 'x'
    var_29 = 'y'
    var_30 = 'tuple'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = module_1.Constant()
    var_34 = module_1.Constant()
    var_35 = [var_33, var_34]
    var_36 = module_1.Load()
    var_37 = module_1.Tuple()
    var_38 = 0
    var_39 = 3
    var_40 = module_1.Constant()



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_0.Parser()
    var_10 = 'ANOTHER_CONST'
    var_11 = 'hello'
    var_12 = module_1.Constant()
    var_13 = 'str'
    var_14 = module_0.Parser()
    var_15 = 'NO_TYPE_CONST'
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = module_0.Parser()
    var_19 = '__all__'
    var_20 = module_1.Constant()
    var_21 = 'func'
    var_22 = module_1.Constant()
    var_23 = [var_20, var_22]
    var_24 = module_1.List()
    var_25 = module_0.Parser()
    var_26 = 'some_list'
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = module_0.Parser()
    var_30 = 'a'
    var_31 = 'b'
    var_32 = module_1.Constant()
    var_33 = module_0.Parser()
    var_34 = 'mod'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = 'name'
    var_38 = module_1.Constant()



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'TEST_CONST_2'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = 'TEST_CONST_3'
    var_14 = 3.14
    var_15 = module_1.Constant()
    var_16 = 'list'
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = 0
    var_20 = module_1.Constant()
    var_21 = module_1.Constant()
    var_22 = '__all__'
    var_23 = module_1.Constant()
    var_24 = [var_23]
    var_25 = module_1.Load()
    var_26 = module_1.Tuple()



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'object'
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = 42
    var_7 = 'attr2'
    var_8 = 'str'
    var_9 = 'hello'
    var_10 = 'attr3'
    var_11 = 3.14
    var_12 = var_0.doc[var_2]
    var_13 = 'Members'
    var_14 = 'Type'
    var_15 = module_0.code(var_7)
    var_16 = module_0.code(var_8)
    var_17 = (var_15, var_16)
    var_18 = module_0.code(var_10)
    var_19 = 'float'
    var_20 = module_0.code(var_19)
    var_21 = (var_18, var_20)
    var_22 = [var_17, var_21]



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'test_root'
    var_5 = 'operating_system'
    var_6 = 'submodule'
    var_7 = 'func'
    var_8 = 1
    var_9 = 'test_root.module'
    var_10 = 'f'
    var_11 = 0



# Parsed testcases at query #11
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
    var_6 = 'not_a_valid_name'
    var_7 = module_1.Constant()
    var_8 = var_2.visit_Constant(var_7)
    var_9 = 'valid_name'
    var_10 = module_1.Constant()
    var_11 = var_2.visit_Constant(var_10)
    var_12 = var_11.ctx
    var_13 = 'invalid syntax'
    var_14 = module_1.Constant()
    var_15 = var_2.visit_Constant(var_14)



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_0.Parser()
    var_10 = 'ANOTHER_CONST'
    var_11 = 'hello'
    var_12 = module_1.Constant()
    var_13 = 'str'
    var_14 = module_0.Parser()
    var_15 = 'NO_TYPE_CONST'
    var_16 = True
    var_17 = module_1.Constant()
    var_18 = module_0.Parser()
    var_19 = '__all__'
    var_20 = 'PUBLIC_FUNC'
    var_21 = module_1.Constant()
    var_22 = 'PublicClass'
    var_23 = module_1.Constant()
    var_24 = [var_21, var_23]
    var_25 = module_1.Load()
    var_26 = module_1.List()
    var_27 = module_0.Parser()
    var_28 = var_27.alias
    var_29 = len(var_28)
    var_30 = 'some_list'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = 0
    var_34 = module_1.Constant()
    var_35 = module_1.Constant()
    var_36 = var_27.alias
    var_37 = len(var_36)
    var_38 = module_0.Parser()
    var_39 = var_38.alias
    var_40 = len(var_39)
    var_41 = 'first'
    var_42 = 'second'
    var_43 = module_1.Constant()
    var_44 = var_38.alias
    var_45 = len(var_44)



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'arg1'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = 'arg2'
    var_6 = module_1.arg()
    var_7 = 'return'
    var_8 = module_1.arg()
    var_9 = [var_4, var_6, var_8]
    var_10 = False
    var_11 = var_0.func_ann(var_1, var_9, has_self=var_10, cls_method=var_10)
    var_12 = list(var_11)
    var_13 = 'self'
    var_14 = module_1.arg()
    var_15 = module_1.arg()
    var_16 = module_1.arg()
    var_17 = [var_14, var_15, var_16]
    var_18 = True
    var_19 = var_0.func_ann(var_1, var_17, has_self=var_18, cls_method=var_10)
    var_20 = list(var_19)
    var_21 = 'cls'
    var_22 = module_1.arg()
    var_23 = module_1.arg()
    var_24 = module_1.arg()
    var_25 = [var_22, var_23, var_24]
    var_26 = var_0.func_ann(var_1, var_25, has_self=var_18, cls_method=var_18)
    var_27 = list(var_26)
    var_28 = 'int'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = module_1.arg()
    var_32 = 'str'
    var_33 = module_1.Load()
    var_34 = module_1.Name()
    var_35 = module_1.arg()
    var_36 = 'bool'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = module_1.arg()
    var_40 = [var_31, var_35, var_39]
    var_41 = var_0.func_ann(var_1, var_40, has_self=var_18, cls_method=var_10)
    var_42 = list(var_41)
    var_43 = module_1.arg()
    var_44 = '*'
    var_45 = module_1.arg()
    var_46 = module_1.arg()
    var_47 = module_1.arg()
    var_48 = [var_43, var_45, var_46, var_47]
    var_49 = var_0.func_ann(var_1, var_48, has_self=var_18, cls_method=var_10)
    var_50 = list(var_49)
    var_51 = 'MyClass'
    var_52 = module_1.Load()
    var_53 = module_1.Name()
    var_54 = module_1.arg()
    var_55 = module_1.arg()
    var_56 = module_1.arg()
    var_57 = [var_54, var_55, var_56]
    var_58 = var_0.func_ann(var_1, var_57, has_self=var_18, cls_method=var_10)
    var_59 = list(var_58)



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = 'math'
    var_7 = 'sqrt'
    var_8 = 'square_root'
    var_9 = 0
    var_10 = 'math.sqrt'
    var_11 = 'numpy'
    var_12 = 'array'
    var_13 = 1
    var_14 = 'test_module.numpy.array'



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'str'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_8, var_11]
    var_13 = module_1.Load()
    var_14 = module_1.Tuple()
    var_15 = module_1.Load()
    var_16 = module_1.Subscript()
    var_17 = var_2.visit_Subscript(var_16)
    var_18 = var_17.left
    var_19 = var_17.op
    var_20 = var_17.right
    var_21 = {}
    var_22 = module_0.Resolver(var_0, var_21)
    var_23 = 'Optional'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.Load()
    var_29 = module_1.Subscript()
    var_30 = var_22.visit_Subscript(var_29)
    var_31 = var_30.left
    var_32 = var_30.op
    var_33 = var_30.right
    var_34 = {}
    var_35 = module_0.Resolver(var_0, var_34)
    var_36 = 'List'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.Load()
    var_42 = module_1.Subscript()
    var_43 = var_35.visit_Subscript(var_42)
    var_44 = var_43.value
    var_45 = {}
    var_46 = module_0.Resolver(var_0, var_45)
    var_47 = 'Custom'
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = module_1.Load()
    var_53 = module_1.Subscript()
    var_54 = var_46.visit_Subscript(var_53)
    var_55 = 'root.Union'
    var_56 = 'typing.Union'
    var_57 = {var_55: var_56}
    var_58 = module_0.Resolver(var_0, var_57)
    var_59 = module_1.Load()
    var_60 = module_1.Name()
    var_61 = module_1.Load()
    var_62 = module_1.Name()
    var_63 = module_1.Load()
    var_64 = module_1.Name()
    var_65 = [var_62, var_64]
    var_66 = module_1.Load()
    var_67 = module_1.Tuple()
    var_68 = module_1.Load()
    var_69 = module_1.Subscript()
    var_70 = var_58.visit_Subscript(var_69)
    var_71 = var_70.left
    var_72 = var_70.op
    var_73 = var_70.right



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = 'test'
    var_4 = module_1.Constant()
    var_5 = module_1.Expr()
    var_6 = [var_5]
    var_7 = module_0.walk_body(var_6)
    var_8 = list(var_7)
    var_9 = True
    var_10 = module_1.Constant()
    var_11 = 'if_true'
    var_12 = module_1.Constant()
    var_13 = module_1.Expr()
    var_14 = [var_13]
    var_15 = 'if_false'
    var_16 = module_1.Constant()
    var_17 = module_1.Expr()
    var_18 = [var_17]
    var_19 = module_1.If()
    var_20 = module_1.Constant()
    var_21 = module_1.Expr()
    var_22 = module_1.Constant()
    var_23 = module_1.Expr()
    var_24 = [var_21, var_23]
    var_25 = [var_19]
    var_26 = module_0.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = 'try_body'
    var_29 = module_1.Constant()
    var_30 = module_1.Expr()
    var_31 = [var_30]
    var_32 = []
    var_33 = 'try_orelse'
    var_34 = module_1.Constant()
    var_35 = module_1.Expr()
    var_36 = [var_35]
    var_37 = 'try_finalbody'
    var_38 = module_1.Constant()
    var_39 = module_1.Expr()
    var_40 = [var_39]
    var_41 = module_1.Try()
    var_42 = module_1.Constant()
    var_43 = module_1.Expr()
    var_44 = module_1.Constant()
    var_45 = module_1.Expr()
    var_46 = module_1.Constant()
    var_47 = module_1.Expr()
    var_48 = [var_43, var_45, var_47]
    var_49 = [var_41]
    var_50 = module_0.walk_body(var_49)
    var_51 = list(var_50)
    var_52 = 'first'
    var_53 = module_1.Constant()
    var_54 = module_1.Expr()
    var_55 = 'last'
    var_56 = module_1.Constant()
    var_57 = module_1.Expr()
    var_58 = [var_54, var_19, var_41, var_57]
    var_59 = module_1.Constant()
    var_60 = module_1.Expr()
    var_61 = module_1.Constant()
    var_62 = module_1.Expr()
    var_63 = module_1.Constant()
    var_64 = module_1.Expr()
    var_65 = module_1.Constant()
    var_66 = module_1.Expr()
    var_67 = module_1.Constant()
    var_68 = module_1.Expr()
    var_69 = module_1.Constant()
    var_70 = module_1.Expr()
    var_71 = module_1.Constant()
    var_72 = module_1.Expr()
    var_73 = [var_60, var_62, var_64, var_66, var_68, var_70, var_72]
    var_74 = module_0.walk_body(var_58)
    var_75 = list(var_74)



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = 'root.name'
    var_6 = 'other_name'
    var_7 = {var_5: var_6}
    var_8 = ''
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = 'name'
    var_11 = module_1.Load()
    var_12 = {var_5: var_5}
    var_13 = module_0.Resolver(var_0, var_12, var_8)
    var_14 = module_1.Load()
    var_15 = 'root.TypeVar'
    var_16 = 'typing.TypeVar'
    var_17 = {var_15: var_16}
    var_18 = module_0.Resolver(var_0, var_17, var_8)
    var_19 = 'TypeVar'
    var_20 = module_1.Load()
    var_21 = {}
    var_22 = module_0.Resolver(var_0, var_21, var_8)
    var_23 = module_1.Load()



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test'
    var_2 = {}
    var_3 = ''
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = var_4.visit_Constant(var_6)
    var_8 = 'not a valid name'
    var_9 = module_1.Constant()
    var_10 = var_4.visit_Constant(var_9)
    var_11 = 'valid_name'
    var_12 = module_1.Constant()
    var_13 = var_4.visit_Constant(var_12)
    var_14 = var_13.ctx
    var_15 = 'invalid name with space'
    var_16 = module_1.Constant()
    var_17 = var_4.visit_Constant(var_16)



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = 'root'
    var_8 = 'y'
    var_9 = 100
    var_10 = module_1.Constant()
    var_11 = 'z'
    var_12 = 3.14
    var_13 = module_1.Constant()
    var_14 = 'float'
    var_15 = 'a'
    var_16 = module_1.Constant()
    var_17 = 'b'
    var_18 = module_1.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_1.Load()
    var_21 = module_1.Tuple()
    var_22 = '__all__'
    var_23 = 1
    var_24 = module_1.Constant()
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = 0
    var_28 = module_1.Constant()
    var_29 = module_1.Constant()
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_1.Constant()
    var_33 = module_1.Load()
    var_34 = module_1.Name()
    var_35 = module_1.Constant()
    var_36 = module_1.Constant()



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'object'
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = 'attr2'
    var_7 = 42
    var_8 = 'enum.Enum'
    var_9 = 'attr3'
    var_10 = 'str'
    var_11 = 'attr4'
    var_12 = 'value'



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = 'attr1'
    var_3 = 'int'
    var_4 = 1
    var_5 = 'attr2'
    var_6 = 'str'
    var_7 = 'test_module'
    var_8 = 'test_module.ClassA'



# Parsed testcases at query #22
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'non_typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'attr'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)
    var_10 = 'typing'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = module_1.Load()
    var_14 = module_1.Attribute()
    var_15 = module_1.Load()
    var_16 = module_1.Name()
    var_17 = var_2.visit_Attribute(var_14)
    var_18 = module_1.Load()
    var_19 = module_1.Name()
    var_20 = 'nested'
    var_21 = module_1.Load()
    var_22 = module_1.Attribute()
    var_23 = module_1.Load()
    var_24 = module_1.Attribute()
    var_25 = var_2.visit_Attribute(var_24)
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.Load()
    var_29 = module_1.Attribute()
    var_30 = module_1.Load()
    var_31 = module_1.Attribute()
    var_32 = var_2.visit_Attribute(var_31)
    var_33 = module_1.Load()



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'example_module'
    var_2 = 'SOME_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'ANOTHER_CONST'
    var_10 = 3.14
    var_11 = module_1.Constant()
    var_12 = 'float'
    var_13 = '__all__'
    var_14 = module_1.Constant()
    var_15 = module_1.Constant()
    var_16 = [var_14, var_15]
    var_17 = module_1.Load()
    var_18 = module_1.Tuple()
    var_19 = module_1.Constant()
    var_20 = 'module'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = 'attr'
    var_24 = 'value'
    var_25 = module_1.Constant()
    var_26 = '42'
    var_27 = '3.14'



# Parsed testcases at query #24
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = True
    var_4 = module_0.Constant()
    var_5 = 'y'
    var_6 = 10
    var_7 = module_0.Constant()
    var_8 = 'z'
    var_9 = 20
    var_10 = module_0.Constant()
    var_11 = 'a'
    var_12 = module_0.Constant()
    var_13 = []
    var_14 = 'b'
    var_15 = 2
    var_16 = module_0.Constant()
    var_17 = 'c'
    var_18 = 3
    var_19 = module_0.Constant()
    var_20 = False
    var_21 = module_0.Constant()
    var_22 = 'd'
    var_23 = 4
    var_24 = module_0.Constant()
    var_25 = []
    var_26 = module_1.walk_body(var_25)
    var_27 = list(var_26)



# Parsed testcases at query #25
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
    var_9 = 1
    var_10 = module_0.Constant()
    var_11 = 2
    var_12 = module_0.Constant()
    var_13 = [var_10, var_12]
    var_14 = module_0.Load()
    var_15 = module_0.Tuple()
    var_16 = module_1.const_type(var_15)
    assert var_16 == 'tuple[int, int]'
    var_17 = 'a'
    var_18 = module_0.Constant()
    var_19 = 'b'
    var_20 = module_0.Constant()
    var_21 = [var_18, var_20]
    var_22 = module_0.Load()
    var_23 = module_0.Tuple()
    var_24 = module_1.const_type(var_23)
    assert var_24 == 'tuple[str, str]'
    var_25 = module_0.Constant()
    var_26 = module_0.Constant()
    var_27 = [var_25, var_26]
    var_28 = module_0.Load()
    var_29 = module_0.Tuple()
    var_30 = module_1.const_type(var_29)
    assert var_30 == 'tuple[Any, Any]'
    var_31 = module_0.Constant()
    var_32 = module_0.Constant()
    var_33 = [var_31, var_32]
    var_34 = module_0.Load()
    var_35 = module_0.List()
    var_36 = module_1.const_type(var_35)
    assert var_36 == 'list[int, int]'
    var_37 = module_0.Constant()
    var_38 = module_0.Constant()
    var_39 = [var_37, var_38]
    var_40 = module_0.Load()
    var_41 = module_0.List()
    var_42 = module_1.const_type(var_41)
    assert var_42 == 'list[str, str]'
    var_43 = module_0.Constant()
    var_44 = module_0.Constant()
    var_45 = [var_43, var_44]
    var_46 = module_0.Load()
    var_47 = module_0.List()
    var_48 = module_1.const_type(var_47)
    assert var_48 == 'list[Any, Any]'
    var_49 = module_0.Constant()
    var_50 = module_0.Constant()
    var_51 = [var_49, var_50]
    var_52 = module_0.Load()
    var_53 = module_0.Set()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'set[int, int]'
    var_55 = module_0.Constant()
    var_56 = module_0.Constant()
    var_57 = [var_55, var_56]
    var_58 = module_0.Load()
    var_59 = module_0.Set()
    var_60 = module_1.const_type(var_59)
    assert var_60 == 'set[str, str]'
    var_61 = module_0.Constant()
    var_62 = module_0.Constant()
    var_63 = [var_61, var_62]
    var_64 = module_0.Load()
    var_65 = module_0.Set()
    var_66 = module_1.const_type(var_65)
    assert var_66 == 'set[Any, Any]'
    var_67 = module_0.Constant()
    var_68 = module_0.Constant()
    var_69 = [var_67, var_68]
    var_70 = module_0.Constant()
    var_71 = module_0.Constant()
    var_72 = [var_70, var_71]
    var_73 = module_0.Load()
    var_74 = module_0.Dict()
    var_75 = module_1.const_type(var_74)
    assert var_75 == 'dict[int, str]'
    var_76 = module_0.Constant()
    var_77 = module_0.Constant()
    var_78 = [var_76, var_77]
    var_79 = module_0.Constant()
    var_80 = module_0.Constant()
    var_81 = [var_79, var_80]
    var_82 = module_0.Load()
    var_83 = module_0.Dict()
    var_84 = module_1.const_type(var_83)
    assert var_84 == 'dict[Any, str]'
    var_85 = module_0.Constant()
    var_86 = module_0.Constant()
    var_87 = [var_85, var_86]
    var_88 = module_0.Constant()
    var_89 = module_0.Constant()
    var_90 = [var_88, var_89]
    var_91 = module_0.Load()
    var_92 = module_0.Dict()
    var_93 = module_1.const_type(var_92)
    assert var_93 == 'dict[str, int]'
    var_94 = 'int'
    var_95 = module_0.Load()
    var_96 = module_0.Name()
    var_97 = []
    var_98 = []
    var_99 = module_0.Call(*var_97)
    var_100 = module_1.const_type(var_99)
    assert var_100 == 'int'
    var_101 = 'str'
    var_102 = module_0.Load()
    var_103 = module_0.Name()
    var_104 = []
    var_105 = []
    var_106 = module_0.Call(*var_104)
    var_107 = module_1.const_type(var_106)
    assert var_107 == 'str'
    var_108 = 'list'
    var_109 = module_0.Load()
    var_110 = module_0.Name()
    var_111 = []
    var_112 = []
    var_113 = module_0.Call(*var_111)
    var_114 = module_1.const_type(var_113)
    assert var_114 == 'list'
    var_115 = module_0.Constant()
    var_116 = module_0.BitOr()
    var_117 = module_0.Constant()
    var_118 = module_0.BinOp()
    var_119 = module_1.const_type(var_118)
    assert var_119 == 'Any'
    var_120 = 'obj'
    var_121 = module_0.Load()
    var_122 = module_0.Name()
    var_123 = 'attr'
    var_124 = module_0.Load()
    var_125 = module_0.Attribute()
    var_126 = module_1.const_type(var_125)
    assert var_126 == 'Any'



# Parsed testcases at query #26
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = 'ANOTHER_CONST'
    var_10 = 'hello'
    var_11 = module_1.Constant()
    var_12 = 'str'
    var_13 = '__all__'
    var_14 = module_1.Constant()
    var_15 = module_1.Constant()
    var_16 = [var_14, var_15]
    var_17 = module_1.Load()
    var_18 = module_1.Tuple()
    var_19 = 'NOT_ALL'
    var_20 = 123
    var_21 = module_1.Constant()
    var_22 = 'A'
    var_23 = 'B'
    var_24 = module_1.Constant()
    var_25 = 'test'
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = 0
    var_29 = module_1.Constant()
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_1.Constant()



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'example_module'
    var_2 = 'example_module.example_function'
    var_3 = 'arg1'
    var_4 = None
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'arg2'
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = '*args'
    var_11 = module_1.arg()
    var_12 = 'kwarg1'
    var_13 = module_1.arg()
    var_14 = [var_13]
    var_15 = '**kwargs'
    var_16 = module_1.arg()
    var_17 = []
    var_18 = []
    var_19 = module_1.arguments(*var_9)
    var_20 = None
    var_21 = False
    var_22 = False
    var_23 = var_0.func_api(var_1, var_2, var_19, var_20, has_self=var_21, cls_method=var_22)
    var_24 = '### example_function()\n\n*Full name:* `example_module.example_function`\n\n'
    var_25 = '| arg1 | arg2 | *args | kwarg1 | **kwargs | return |\n'
    var_26 = var_24 + var_25
    var_27 = '|------|------|-------|--------|----------|--------|\n'
    var_28 = var_26 + var_27
    var_29 = '| Any | Any | Any | Any | Any | Any |\n'
    var_30 = var_28 + var_29



# Parsed testcases at query #29
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'base_class'
    var_4 = 'attr1'
    var_5 = 'int'
    var_6 = None
    var_7 = 'attr2'
    var_8 = 'str'
    var_9 = 'attr3'
    var_10 = 'float'
    var_11 = 'attr4'
    var_12 = 'bool'
    var_13 = module_0.Parser()
    var_14 = 'test_module'
    var_15 = 'test_module.TestEnum'
    var_16 = 'enum.Enum'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.module.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = '_private.module.name'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is False
    var_4 = 'public._private.module'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is False
    var_6 = 'public.module.__magic__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = '_private.module.__magic__'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is False
    var_10 = 'public'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_12 = '_private'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is False
    var_14 = ''
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is True
    var_16 = 'public.module.submodule.name'
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = 'public.module._submodule.name'
    var_19 = module_0.is_public_family(var_18)
    assert var_19 is False



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'var_name'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = module_1.Constant()
    var_9 = '__all__'
    var_10 = 'func1'
    var_11 = module_1.Constant()
    var_12 = 'func2'
    var_13 = module_1.Constant()
    var_14 = [var_11, var_13]
    var_15 = module_1.Load()
    var_16 = module_1.Tuple()
    var_17 = module_1.Constant()
    var_18 = module_1.Constant()
    var_19 = [var_17, var_18]
    var_20 = module_1.Load()
    var_21 = module_1.List()
    var_22 = 'obj'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = 'attr'
    var_26 = module_1.Constant()
    var_27 = 'var1'
    var_28 = 'var2'
    var_29 = module_1.Constant()
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = module_1.Constant()



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = module_0.Parser()
    var_3 = var_2.compile()
    assert var_3 == '# Module `module`\n\n\n'
    var_4 = module_0.Parser()
    var_5 = var_4.compile()
    assert var_5 == '# Module `module`\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n\n\n'
    var_6 = module_0.Parser()
    var_7 = var_6.compile()
    assert var_7 == '# Module `module`\n\n\n## func()\n\n*Full name:* `module.func`\n\n\n'
    var_8 = True
    var_9 = module_0.Parser(toc=var_8)
    var_10 = var_9.compile()
    assert var_10 == '**Table of contents:**\n+ [`module`](#module)\n\n# Module `module`\n\n\n'
    var_11 = module_0.Parser()
    var_12 = var_11.compile()
    assert var_12 == '# Module `module`\n\n\n'
    var_13 = module_0.Parser()
    var_14 = 'module.public'
    var_15 = var_13.compile()
    assert var_15 == '# Module `module`\n\n\n## public()\n\n*Full name:* `module.public`\n\n\n'



# Parsed testcases at query #4
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\n    def test_function():\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '\n    class TestClass:\n        def test_method(self):\n            pass\n    '
    var_5 = var_0.parse(var_1, var_4)
    var_6 = '\n    @staticmethod\n    def test_static_method():\n        pass\n    '
    var_7 = var_0.parse(var_1, var_6)
    var_8 = '\n    @classmethod\n    def test_class_method(cls):\n        pass\n    '
    var_9 = var_0.parse(var_1, var_8)
    var_10 = '\n    class TestClass:\n        @staticmethod\n        def test_static_method():\n            pass\n    '
    var_11 = var_0.parse(var_1, var_10)
    var_12 = '\n    class TestClass:\n        @classmethod\n        def test_class_method(cls):\n            pass\n    '
    var_13 = var_0.parse(var_1, var_12)



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'example'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)
    var_12 = var_11.value
    var_13 = 'Union'
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
    var_28 = 'Optional'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = module_1.Load()
    var_34 = module_1.Subscript()
    var_35 = var_2.visit_Subscript(var_34)
    var_36 = var_35.op
    var_37 = var_35.right
    var_38 = 'Dict'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = module_1.Load()
    var_44 = module_1.Name()
    var_45 = [var_42, var_44]
    var_46 = module_1.Load()
    var_47 = module_1.Tuple()
    var_48 = module_1.Load()
    var_49 = module_1.Subscript()
    var_50 = var_2.visit_Subscript(var_49)
    var_51 = var_50.value



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Union'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'str'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = [var_9, var_12]
    var_14 = module_1.Load()
    var_15 = module_1.Tuple()
    var_16 = module_1.Load()
    var_17 = module_1.Subscript()
    var_18 = var_3.visit_Subscript(var_17)
    var_19 = var_18.left
    var_20 = var_18.op
    var_21 = var_18.right
    var_22 = 'Optional'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = module_1.Load()
    var_28 = module_1.Subscript()
    var_29 = var_3.visit_Subscript(var_28)
    var_30 = var_29.left
    var_31 = var_29.op
    var_32 = var_29.right
    var_33 = 'List'
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = module_1.Load()
    var_37 = module_1.Name()
    var_38 = module_1.Load()
    var_39 = module_1.Subscript()
    var_40 = var_3.visit_Subscript(var_39)
    var_41 = var_40.value
    var_42 = var_40.slice
    var_43 = 'Custom'
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = module_1.Load()
    var_49 = module_1.Subscript()
    var_50 = var_3.visit_Subscript(var_49)
    var_51 = var_50.value
    var_52 = var_50.slice



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'object'
    var_4 = 'attr'
    var_5 = 'int'
    var_6 = None
    var_7 = 'attr2'
    var_8 = 'str'
    var_9 = 'method'
    var_10 = module_1.arguments()
    var_11 = []
    var_12 = []
    var_13 = []



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submodule'
    var_3 = 'module.public_func'
    var_4 = 'module.public_class'
    var_5 = 'module.submodule.public_func'
    var_6 = {var_3, var_4, var_5}
    var_7 = {var_5}
    var_8 = var_0.is_public(var_3)
    assert var_8 is True
    var_9 = var_0.is_public(var_4)
    assert var_9 is True
    var_10 = var_0.is_public(var_5)
    assert var_10 is True
    var_11 = 'module.private_func'
    var_12 = var_0.is_public(var_11)
    assert var_12 is False
    var_13 = 'module.submodule.private_func'
    var_14 = var_0.is_public(var_13)
    assert var_14 is False
    var_15 = var_0.is_public(var_1)
    assert var_15 is True
    var_16 = var_0.is_public(var_2)
    assert var_16 is True
    var_17 = 'non_existent_module'
    var_18 = var_0.is_public(var_17)
    assert var_18 is False
    var_19 = set()
    var_20 = set()
    var_21 = var_0.is_public(var_3)
    assert var_21 is False
    var_22 = var_0.is_public(var_5)
    assert var_22 is False
    var_23 = var_0.is_public(var_1)
    assert var_23 is True
    var_24 = var_0.is_public(var_2)
    assert var_24 is True



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = module_1.arguments()
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = module_1.FunctionDef(*var_3)
    var_8 = var_0.api(var_1, var_7)
    var_9 = 'test_async_func'
    var_10 = module_1.arguments()
    var_11 = []
    var_12 = []
    var_13 = module_1.AsyncFunctionDef(*var_10)
    var_14 = var_0.api(var_1, var_13)
    var_15 = 'TestClass'
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = module_1.ClassDef()
    var_20 = var_0.api(var_1, var_19)
    var_21 = 'test_method'
    var_22 = module_1.arguments()
    var_23 = []
    var_24 = []
    var_25 = module_1.FunctionDef(*var_22)
    var_26 = var_0.api(var_1, var_25, prefix=var_15)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.func'
    var_2 = 'module.Class'
    var_3 = '# func()\n\n*Full name:* `module.func`\n\n'
    var_4 = '# class Class\n\n*Full name:* `module.Class`\n\n'
    var_5 = 'Function documentation.'
    var_6 = 'Class documentation.'
    var_7 = 1
    var_8 = 'module'
    var_9 = set()
    var_10 = '# func()\n\n*Full name:* `module.func`\n\nFunction documentation.\n\n# class Class\n\n*Full name:* `module.Class`\n\nClass documentation.\n\n'
    var_11 = var_0.compile()
    var_12 = '**Table of contents:**\n    + [`module.func`](#module-func)\n    + [`module.Class`](#module-class)\n\n# func()\n\n*Full name:* `module.func`\n\nFunction documentation.\n\n# class Class\n\n*Full name:* `module.Class`\n\nClass documentation.\n\n'
    var_13 = var_0.compile()



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = 'root.SomeType'
    var_8 = 'SomeExpression'
    var_9 = {var_7: var_8}
    var_10 = ''
    var_11 = module_0.Resolver(var_0, var_9, var_10)
    var_12 = 'SomeType'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = var_11.visit_Name(var_14)
    var_16 = module_1.unparse(var_15)
    assert var_16 == 'SomeExpression'
    var_17 = {}
    var_18 = module_0.Resolver(var_0, var_17, var_10)
    var_19 = 'AnotherType'
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = var_18.visit_Name(var_21)
    var_23 = 'root.TypeVar'
    var_24 = 'typing.TypeVar'
    var_25 = {var_23: var_24}
    var_26 = module_0.Resolver(var_0, var_25, var_10)
    var_27 = 'TypeVar'
    var_28 = module_1.Load()
    var_29 = module_1.Name()
    var_30 = var_26.visit_Name(var_29)



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.func'
    var_3 = []
    var_4 = 'arg1'
    var_5 = None
    var_6 = module_1.arg()
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = module_1.arguments(*var_7)
    var_12 = None
    var_13 = False
    var_14 = False
    var_15 = var_0.func_api(var_1, var_2, var_11, var_12, has_self=var_13, cls_method=var_14)
    var_16 = '### func()\n\n*Full name:* `test_module.func`\n\n| arg1 |\n|------|\n| Any  |\n'



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = 'not a valid name'
    var_5 = 'valid_name'
    var_6 = 'a + b'



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'math'
    var_2 = 'm'
    var_3 = 'root'
    var_4 = 'os'
    var_5 = 'path'
    var_6 = None
    var_7 = 1
    var_8 = 'root.module'
    var_9 = 'sys'
    var_10 = 'version'
    var_11 = 'ver'
    var_12 = 0
    var_13 = 'o'



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = '# Module `module.name`\n\n'
    var_3 = 'Mock docstring'
    var_4 = 'module'



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'typing.Union'
    var_1 = 'typing.Optional'
    var_2 = 'typing.List'
    var_3 = 'Union'
    var_4 = 'Optional'
    var_5 = 'list'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'typing'
    var_8 = module_0.Resolver(var_7, var_6)
    var_9 = module_1.Load()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = 'str'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = module_1.Load()
    var_16 = module_0.Resolver(var_7, var_6)
    var_17 = module_1.Load()
    var_18 = module_1.Load()
    var_19 = module_1.Load()
    var_20 = module_0.Resolver(var_7, var_6)
    var_21 = 'List'
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = module_1.Load()
    var_25 = module_0.Resolver(var_7, var_6)
    var_26 = 'Dict'
    var_27 = module_1.Load()
    var_28 = module_1.Load()
    var_29 = module_1.Load()
    var_30 = module_1.Load()
    var_31 = module_1.Load()



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_VAR'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 1
    var_9 = module_0.Parser()
    var_10 = 'ANOTHER_VAR'
    var_11 = 'hello'
    var_12 = module_1.Constant()
    var_13 = 'str'
    var_14 = module_0.Parser()
    var_15 = 'NO_TYPE_VAR'
    var_16 = 3.14
    var_17 = module_1.Constant()
    var_18 = module_0.Parser()
    var_19 = '__all__'
    var_20 = 'public_func'
    var_21 = module_1.Constant()
    var_22 = 'PublicClass'
    var_23 = module_1.Constant()
    var_24 = [var_21, var_23]
    var_25 = module_1.Load()
    var_26 = module_1.List()
    var_27 = module_0.Parser()
    var_28 = 'var1'
    var_29 = 'var2'
    var_30 = module_1.Constant()
    var_31 = module_0.Parser()
    var_32 = 'obj'
    var_33 = module_1.Load()
    var_34 = module_1.Name()
    var_35 = 'attr'
    var_36 = module_1.Constant()



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'module_name'
    var_3 = 'alias_name'
    var_4 = module_0.Parser()
    var_5 = 'name'
    var_6 = 1
    var_7 = module_0.Parser()
    var_8 = 0
    var_9 = module_0.Parser()
    var_10 = 'module1'
    var_11 = 'alias1'
    var_12 = 'module2'
    var_13 = 'alias2'



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = 'test'
    var_4 = module_1.Constant()
    var_5 = module_1.Expr()
    var_6 = [var_5]
    var_7 = module_0.walk_body(var_6)
    var_8 = list(var_7)
    var_9 = True
    var_10 = module_1.Constant()
    var_11 = 'if body'
    var_12 = module_1.Constant()
    var_13 = module_1.Expr()
    var_14 = [var_13]
    var_15 = 'else body'
    var_16 = module_1.Constant()
    var_17 = module_1.Expr()
    var_18 = [var_17]
    var_19 = module_1.If()
    var_20 = [var_19]
    var_21 = module_1.Constant()
    var_22 = module_1.Expr()
    var_23 = module_1.Constant()
    var_24 = module_1.Expr()
    var_25 = [var_22, var_24]
    var_26 = module_0.walk_body(var_20)
    var_27 = list(var_26)
    var_28 = 'try body'
    var_29 = module_1.Constant()
    var_30 = module_1.Expr()
    var_31 = [var_30]
    var_32 = 'handler body'
    var_33 = module_1.Constant()
    var_34 = module_1.Expr()
    var_35 = [var_34]
    var_36 = module_1.arg()
    var_37 = [var_36]
    var_38 = module_1.Constant()
    var_39 = module_1.Expr()
    var_40 = [var_39]
    var_41 = 'final body'
    var_42 = module_1.Constant()
    var_43 = module_1.Expr()
    var_44 = [var_43]
    var_45 = module_1.Try()
    var_46 = [var_45]
    var_47 = module_1.Constant()
    var_48 = module_1.Expr()
    var_49 = module_1.Constant()
    var_50 = module_1.Expr()
    var_51 = module_1.Constant()
    var_52 = module_1.Expr()
    var_53 = module_1.Constant()
    var_54 = module_1.Expr()
    var_55 = [var_48, var_50, var_52, var_54]
    var_56 = module_0.walk_body(var_46)
    var_57 = list(var_56)
    var_58 = module_1.Constant()
    var_59 = 'outer if'
    var_60 = module_1.Constant()
    var_61 = module_1.Expr()
    var_62 = False
    var_63 = module_1.Constant()
    var_64 = 'inner if'
    var_65 = module_1.Constant()
    var_66 = module_1.Expr()
    var_67 = [var_66]
    var_68 = []
    var_69 = module_1.If()
    var_70 = [var_61, var_69]
    var_71 = []
    var_72 = module_1.If()
    var_73 = [var_72]
    var_74 = module_1.Constant()
    var_75 = module_1.Expr()
    var_76 = module_1.Constant()
    var_77 = module_1.Expr()
    var_78 = [var_75, var_77]
    var_79 = module_0.walk_body(var_73)
    var_80 = list(var_79)
    var_81 = 'simple'
    var_82 = module_1.Constant()
    var_83 = module_1.Expr()
    var_84 = module_1.Constant()
    var_85 = module_1.Constant()
    var_86 = module_1.Expr()
    var_87 = [var_86]
    var_88 = []
    var_89 = module_1.If()
    var_90 = module_1.Constant()
    var_91 = module_1.Expr()
    var_92 = [var_91]
    var_93 = []
    var_94 = []
    var_95 = []
    var_96 = module_1.Try()
    var_97 = [var_83, var_89, var_96]
    var_98 = module_1.Constant()
    var_99 = module_1.Expr()
    var_100 = module_1.Constant()
    var_101 = module_1.Expr()
    var_102 = module_1.Constant()
    var_103 = module_1.Expr()
    var_104 = [var_99, var_101, var_103]
    var_105 = module_0.walk_body(var_97)
    var_106 = list(var_105)



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submodule'
    var_3 = 'module.submodule.func'
    var_4 = 'module.submodule.Class'
    var_5 = 'module.submodule.Class.method'
    var_6 = {var_2, var_3}
    var_7 = {var_4}
    var_8 = var_0.is_public(var_1)
    assert var_8 is True
    var_9 = var_0.is_public(var_2)
    assert var_9 is True
    var_10 = var_0.is_public(var_3)
    assert var_10 is True
    var_11 = var_0.is_public(var_4)
    assert var_11 is True
    var_12 = var_0.is_public(var_5)
    assert var_12 is True
    var_13 = 'module.private_func'
    var_14 = var_0.is_public(var_13)
    assert var_14 is False
    var_15 = 'module.submodule.private_method'
    var_16 = var_0.is_public(var_15)
    assert var_16 is False
    var_17 = var_0.is_public(var_3)
    assert var_17 is False
    var_18 = var_0.is_public(var_4)
    assert var_18 is False
    var_19 = 'module.public_func'
    var_20 = var_0.is_public(var_19)
    assert var_20 is True
    var_21 = 'module.submodule.public_method'
    var_22 = var_0.is_public(var_21)
    assert var_22 is True
    var_23 = 'module.MixedCase'
    var_24 = var_0.is_public(var_23)
    assert var_24 is True
    var_25 = 'module.submodule.MixedCaseMethod'
    var_26 = var_0.is_public(var_25)
    assert var_26 is True



# Parsed testcases at query #22
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = module_1.arg()
    var_9 = 'y'
    var_10 = 'str'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = module_1.arg()
    var_14 = '*args'
    var_15 = 'Any'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = module_1.arg()
    var_19 = '**kwargs'
    var_20 = 'Dict'
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = module_1.arg()
    var_24 = 'return'
    var_25 = 'None'
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.arg()
    var_29 = [var_3, var_8, var_13, var_18, var_23, var_28]
    var_30 = 'root'
    var_31 = True
    var_32 = False
    var_33 = var_0.func_ann(var_30, var_29, has_self=var_31, cls_method=var_32)
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
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = module_1.arg()
    var_49 = module_1.Load()
    var_50 = module_1.Name()
    var_51 = module_1.arg()
    var_52 = module_1.Load()
    var_53 = module_1.Name()
    var_54 = module_1.arg()
    var_55 = [var_39, var_42, var_45, var_48, var_51, var_54]
    var_56 = var_0.func_ann(var_30, var_55, has_self=var_31, cls_method=var_31)
    var_57 = list(var_56)
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = module_1.arg()
    var_61 = module_1.Load()
    var_62 = module_1.Name()
    var_63 = module_1.arg()
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = module_1.arg()
    var_67 = module_1.Load()
    var_68 = module_1.Name()
    var_69 = module_1.arg()
    var_70 = module_1.Load()
    var_71 = module_1.Name()
    var_72 = module_1.arg()
    var_73 = [var_60, var_63, var_66, var_69, var_72]
    var_74 = var_0.func_ann(var_30, var_73, has_self=var_32, cls_method=var_32)
    var_75 = list(var_74)



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = 'not a valid name'
    var_5 = 'valid_name'
    var_6 = 'invalid name with space'
    var_7 = 'a + b'



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom typing import List, Dict\n\ndef example_function(param: List[str]) -> Dict[str, int]:\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'Missing documentation for test_module.example_function'
    var_5 = var_0.compile()
    var_6 = var_4 in var_5



# Parsed testcases at query #25
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a simple string.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a simple string.'
    var_2 = ">>> print('Hello, World!')\nHello, World!"
    var_3 = "```python\n>>> print('Hello, World!')\nHello, World!\n```"
    var_4 = module_0.doctest(var_2)
    var_5 = ">>> print('Hello')\nHello\n>>> print('World!')\nWorld!"
    var_6 = "```python\n>>> print('Hello')\nHello\n```\n```python\n>>> print('World!')\nWorld!\n```"
    var_7 = module_0.doctest(var_5)
    var_8 = "This is a test.\n>>> print('Hello')\nHello\nThis is another test."
    var_9 = "This is a test.\n```python\n>>> print('Hello')\nHello\n```\nThis is another test."
    var_10 = module_0.doctest(var_8)
    var_11 = '>>> a = 1\n>>> b = 2\n>>> print(a + b)\n3'
    var_12 = '```python\n>>> a = 1\n>>> b = 2\n>>> print(a + b)\n3\n```'
    var_13 = module_0.doctest(var_11)
    var_14 = ''
    var_15 = module_0.doctest(var_14)
    assert var_15 == ''
    var_16 = ">>> print('Hello')\n>>> print('World')"
    var_17 = "```python\n>>> print('Hello')\n```\n```python\n>>> print('World')\n```"
    var_18 = module_0.doctest(var_16)



