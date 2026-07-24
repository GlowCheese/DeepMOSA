####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\nclass BaseClass:\n    pass\n\nclass DerivedClass(BaseClass):\n    x: int\n    y: str = "default"\n    _private: float\n    \n    def method(self):\n        pass\n'
    var_4 = module_1.parse(var_3)
    var_5 = 1
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.DerivedClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_1.class_api(var_2, var_7, var_8, var_9)
    var_11 = var_1.doc[var_7]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api with enum class.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.enum'
    var_4 = 'enum'
    var_5 = '\nimport enum\n\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_6 = module_1.parse(var_5)
    var_7 = 1
    var_8 = var_6.body[var_7]
    var_9 = 'test_module.Color'
    var_10 = var_8.bases
    var_11 = var_8.body
    var_12 = var_1.class_api(var_2, var_9, var_10, var_11)
    var_13 = var_1.doc[var_9]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api with class members.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\nclass MyClass:\n    count: int\n    name: str\n    _internal: bool\n    value: float = 3.14\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_1.class_api(var_2, var_7, var_8, var_9)
    var_11 = var_1.doc[var_7]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api with deleted members.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\nclass MyClass:\n    x: int\n    y: str\n    del x\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_1.class_api(var_2, var_7, var_8, var_9)
    var_11 = var_1.doc[var_7]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api with empty class.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'class EmptyClass: pass'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.EmptyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_1.class_api(var_2, var_7, var_8, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api with type comments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\nclass MyClass:\n    value = 42  # type: int\n    name = "test"  # type: str\n'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_1.class_api(var_2, var_7, var_8, var_9)
    var_11 = var_1.doc[var_7]



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = True
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'int'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'typing.List'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'not a valid python expression !!!'
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = ''
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = '(invalid'
    var_29 = module_1.Constant()
    var_30 = var_3.visit_Constant(var_29)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test the doctest function.'
    var_1 = ''
    var_2 = module_0.doctest(var_1)
    assert var_2 == ''
    var_3 = 'This is a comment'
    var_4 = module_0.doctest(var_3)
    assert var_4 == 'This is a comment'
    var_5 = ">>> print('hello')"
    var_6 = module_0.doctest(var_5)
    assert var_6 == "```python\n>>> print('hello')\n```"
    var_7 = '>>> x = 1\n>>> print(x)'
    var_8 = module_0.doctest(var_7)
    assert var_8 == '```python\n>>> x = 1\n>>> print(x)\n```'
    var_9 = ">>> print('hello')\nhello"
    var_10 = module_0.doctest(var_9)
    assert var_10 == "```python\n>>> print('hello')\nhello\n```"
    var_11 = '>>> x = 1\nsome comment\n>>> y = 2'
    var_12 = module_0.doctest(var_11)
    assert var_12 == '```python\n>>> x = 1\n```\nsome comment\n```python\n>>> y = 2\n```'
    var_13 = "Some comment\n>>> print('end')"
    var_14 = module_0.doctest(var_13)
    assert var_14 == "Some comment\n```python\n>>> print('end')\n```"
    var_15 = '>>> a = 1\n>>> b = 2\n>>> c = a + b'
    var_16 = module_0.doctest(var_15)
    assert var_16 == '```python\n>>> a = 1\n>>> b = 2\n>>> c = a + b\n```'
    var_17 = '>>> x = 5\n5\n>>> y = 10'
    var_18 = module_0.doctest(var_17)
    assert var_18 == '```python\n>>> x = 5\n5\n```\n```python\n>>> y = 10\n```'
    var_19 = '>>> x = 1\ncomment line\n>>> y = 2'
    var_20 = module_0.doctest(var_19)
    assert var_20 == '```python\n>>> x = 1\n```\ncomment line\n```python\n>>> y = 2\n```'
    var_21 = 'output line 1\noutput line 2'
    var_22 = module_0.doctest(var_21)
    assert var_22 == 'output line 1\noutput line 2'



# Parsed testcases at query #4
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test walk_body function.'
    var_1 = []
    var_2 = module_0.walk_body(var_1)
    var_3 = list(var_2)
    var_4 = 'x = 1'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body
    var_7 = module_0.walk_body(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_13 = module_1.parse(var_12)
    var_14 = var_13.body
    var_15 = module_0.walk_body(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_19 = module_1.parse(var_18)
    var_20 = var_19.body
    var_21 = module_0.walk_body(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3'
    var_25 = module_1.parse(var_24)
    var_26 = var_25.body
    var_27 = module_0.walk_body(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 3
    var_30 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3'
    var_31 = module_1.parse(var_30)
    var_32 = var_31.body
    var_33 = module_0.walk_body(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 3
    var_36 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\nelse:\n    z = 3'
    var_37 = module_1.parse(var_36)
    var_38 = var_37.body
    var_39 = module_0.walk_body(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 3
    var_42 = 'x = 1\ny = 2\nz = 3'
    var_43 = module_1.parse(var_42)
    var_44 = var_43.body
    var_45 = module_0.walk_body(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = 'if True:\n    x = 1\n    y = 2\nelse:\n    z = 3\n    w = 4'
    var_49 = module_1.parse(var_48)
    var_50 = var_49.body
    var_51 = module_0.walk_body(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 4
    var_54 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept TypeError:\n    z = 3'
    var_55 = module_1.parse(var_54)
    var_56 = var_55.body
    var_57 = module_0.walk_body(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 3
    var_60 = 'print(1)'
    var_61 = module_1.parse(var_60)
    var_62 = var_61.body
    var_63 = module_0.walk_body(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = var_64[var_10]
    var_67 = 'if True:\n    if True:\n        if True:\n            x = 1'
    var_68 = module_1.parse(var_67)
    var_69 = var_68.body
    var_70 = module_0.walk_body(var_69)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 1
    var_73 = var_71[var_10]



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'TypeAlias'
    var_4 = 'type'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 1
    var_11 = 'var'
    var_12 = 'int'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = None
    var_16 = 'CONSTANT'
    var_17 = 42
    var_18 = module_1.Constant()
    var_19 = 'value'
    var_20 = 10
    var_21 = module_1.Constant()
    var_22 = 'MAX_VALUE'
    var_23 = 100
    var_24 = module_1.Constant()
    var_25 = '__all__'
    var_26 = 'func1'
    var_27 = module_1.Constant()
    var_28 = 'func2'
    var_29 = module_1.Constant()
    var_30 = [var_27, var_29]
    var_31 = module_1.Load()
    var_32 = module_1.Tuple()
    var_33 = 'func3'
    var_34 = module_1.Constant()
    var_35 = [var_34]
    var_36 = module_1.Load()
    var_37 = module_1.List()
    var_38 = 'a'
    var_39 = 'b'
    var_40 = module_1.Constant()
    var_41 = 'obj'
    var_42 = module_1.Load()
    var_43 = module_1.Name()
    var_44 = 'attr'
    var_45 = module_1.Constant()
    var_46 = var_1.alias
    var_47 = len(var_46)
    assert var_47 == 5



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.compile method.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    var_3 = '\n'
    var_4 = True
    var_5 = False
    var_6 = module_0.Parser(var_4, var_4, var_5)
    var_7 = var_6.compile()
    var_8 = module_0.Parser(var_4, var_4, var_4)
    var_9 = var_8.compile()
    var_10 = module_0.Parser(var_5, var_4, var_5)
    var_11 = var_10.compile()
    var_12 = module_0.Parser(var_4, var_4, var_5)
    var_13 = var_12.compile()
    var_14 = module_0.Parser(var_4, var_4, var_4)
    var_15 = var_14.compile()
    var_16 = module_0.Parser(var_5, var_4, var_5)
    var_17 = 'test.public'
    var_18 = var_16.compile()



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test function docstring'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test function docstring'
    var_1 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring with no docstrings.'
    var_1 = 'test_module'
    var_2 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring only processes matching root.'
    var_1 = 'test_module'
    var_2 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring with empty module.'
    var_1 = 'empty_module'
    var_2 = module_0.Parser()



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'def test_func(x: int) -> str: pass'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_1.api(var_2, var_6)
    var_8 = module_0.Parser()
    var_9 = 'async_module'
    var_10 = 'async def async_func(): pass'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_5]
    var_13 = var_8.api(var_9, var_12)
    var_14 = module_0.Parser()
    var_15 = 'class_module'
    var_16 = 'class TestClass: pass'
    var_17 = module_1.parse(var_16)
    var_18 = var_17.body[var_5]
    var_19 = var_14.api(var_15, var_18)
    var_20 = module_0.Parser()
    var_21 = 'nested_module'
    var_22 = 'class OuterClass:\n    def inner_method(self): pass'
    var_23 = module_1.parse(var_22)
    var_24 = var_23.body[var_5]
    var_25 = var_24.body[var_5]
    var_26 = var_20.api(var_21, var_24)
    var_27 = 'OuterClass'
    var_28 = var_20.api(var_21, var_25, prefix=var_27)
    var_29 = module_0.Parser()
    var_30 = 'decorator_module'
    var_31 = '@staticmethod\ndef decorated_func(): pass'
    var_32 = module_1.parse(var_31)
    var_33 = var_32.body[var_5]
    var_34 = var_29.api(var_30, var_33)



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api method of Parser class.'
    var_1 = module_0.Parser()
    var_2 = '\ndef example_func(a: int, b: str = "default", *args, c: float = 1.0, **kwargs) -> bool:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.example_func'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = var_1.func_api(var_6, var_7, var_8, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api with self parameter (instance method).'
    var_1 = module_0.Parser()
    var_2 = '\ndef method(self, x: int) -> str:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.method'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = True
    var_11 = False
    var_12 = var_1.func_api(var_6, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api with classmethod.'
    var_1 = module_0.Parser()
    var_2 = '\ndef create(cls, value: int) -> None:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.create'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = True
    var_11 = var_1.func_api(var_6, var_7, var_8, var_9, has_self=var_10, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api with required arguments only.'
    var_1 = module_0.Parser()
    var_2 = '\ndef simple(x: int, y: str) -> None:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.simple'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = False
    var_11 = False
    var_12 = var_1.func_api(var_6, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = var_1.doc[var_7]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api correctly formats annotations.'
    var_1 = module_0.Parser()
    var_2 = 'test_module.List'
    var_3 = 'typing.List'
    var_4 = '\ndef typed_func(items: list, count: int = 0) -> dict:\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module'
    var_9 = 'test_module.typed_func'
    var_10 = var_7.args
    var_11 = var_7.returns
    var_12 = False
    var_13 = False
    var_14 = var_1.func_api(var_8, var_9, var_10, var_11, has_self=var_12, cls_method=var_13)
    var_15 = var_1.doc[var_9]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api with positional-only arguments.'
    var_1 = module_0.Parser()
    var_2 = '\ndef pos_only(a: int, /, b: str) -> None:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.pos_only'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = False
    var_11 = False
    var_12 = var_1.func_api(var_6, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = var_1.doc[var_7]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api with keyword-only arguments.'
    var_1 = module_0.Parser()
    var_2 = '\ndef kw_only(a: int, *, b: str) -> None:\n    pass\n'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.kw_only'
    var_8 = var_5.args
    var_9 = var_5.returns
    var_10 = False
    var_11 = False
    var_12 = var_1.func_api(var_6, var_7, var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = var_1.doc[var_7]



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'os'
    var_6 = 'List'
    var_7 = 'Optional'
    var_8 = 'MY_CONSTANT'
    var_9 = 'my_function'
    var_10 = 'MyClass'
    var_11 = 'method'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse with __all__ definition.'
    var_1 = module_0.Parser()
    var_2 = 'test_module_all'
    var_3 = "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n"
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'public_func'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse with type comments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module_comment'
    var_3 = '\nx = 42  # type: int\ny = "hello"  # type: str\n'
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'x'
    var_6 = 'y'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse with nested class.'
    var_1 = module_0.Parser()
    var_2 = 'test_nested'
    var_3 = "\nclass Outer:\n    '''Outer class.'''\n    \n    class Inner:\n        '''Inner class.'''\n        \n        def inner_method(self):\n            '''Inner method.'''\n            pass\n"
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'Outer'
    var_6 = 'Inner'
    var_7 = 'inner_method'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse with async function.'
    var_1 = module_0.Parser()
    var_2 = 'test_async'
    var_3 = "\nasync def async_func(x: int) -> str:\n    '''Async function.'''\n    return str(x)\n"
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'async_func'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.parse with decorated functions.'
    var_1 = module_0.Parser()
    var_2 = 'test_decorators'
    var_3 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'MyClass'
    var_6 = 'static_method'
    var_7 = 'class_method'



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.api method.'
    var_1 = module_0.Parser()
    var_2 = "\ndef example_func(x: int) -> str:\n    '''Example function.'''\n    return str(x)\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = module_1.parse(var_2)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_1.api(var_3, var_7)
    var_9 = var_7.name
    var_10 = "\nasync def async_func() -> None:\n    '''Async function.'''\n    pass\n"
    var_11 = 'test_async'
    var_12 = var_1.parse(var_11, var_10)
    var_13 = module_1.parse(var_10)
    var_14 = var_13.body[var_6]
    var_15 = var_1.api(var_11, var_14)
    var_16 = var_14.name
    var_17 = "\nclass ExampleClass:\n    '''Example class.'''\n    pass\n"
    var_18 = 'test_class'
    var_19 = var_1.parse(var_18, var_17)
    var_20 = module_1.parse(var_17)
    var_21 = var_20.body[var_6]
    var_22 = var_1.api(var_18, var_21)
    var_23 = var_21.name
    var_24 = "\nclass Container:\n    def method(self) -> int:\n        '''Nested method.'''\n        return 42\n"
    var_25 = 'test_nested'
    var_26 = var_1.parse(var_25, var_24)
    var_27 = module_1.parse(var_24)
    var_28 = var_27.body[var_6]
    var_29 = var_28.body[var_6]
    var_30 = 'Container'
    var_31 = var_1.api(var_25, var_29, prefix=var_30)
    var_32 = var_29.name
    var_33 = '\n@property\ndef decorated_func() -> str:\n    \'\'\'Decorated function.\'\'\'\n    return "test"\n'
    var_34 = 'test_decorated'
    var_35 = var_1.parse(var_34, var_33)
    var_36 = module_1.parse(var_33)
    var_37 = var_36.body[var_6]
    var_38 = var_1.api(var_34, var_37)
    var_39 = var_37.name



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'MAX_VALUE'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 100
    var_14 = module_1.Constant()
    var_15 = 'y'
    var_16 = 3.14
    var_17 = module_1.Constant()
    var_18 = 'float'
    var_19 = 'z'
    var_20 = module_1.Constant()
    var_21 = 2
    var_22 = module_1.Constant()
    var_23 = [var_20, var_22]
    var_24 = module_1.Load()
    var_25 = module_1.List()
    var_26 = None
    var_27 = '__all__'
    var_28 = 'func1'
    var_29 = module_1.Constant()
    var_30 = 'func2'
    var_31 = module_1.Constant()
    var_32 = [var_29, var_31]
    var_33 = module_1.Load()
    var_34 = module_1.Tuple()
    var_35 = 'a'
    var_36 = 'b'
    var_37 = module_1.Constant()
    var_38 = 'w'
    var_39 = 'str'
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = 'obj'
    var_43 = module_1.Load()
    var_44 = module_1.Name()
    var_45 = 'attr'
    var_46 = module_1.Constant()



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'MyType'
    var_4 = 'str'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'hello'
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'CONSTANT'
    var_11 = 42
    var_12 = module_1.Constant()
    var_13 = None
    var_14 = 'DEBUG'
    var_15 = True
    var_16 = module_1.Constant()
    var_17 = '__all__'
    var_18 = 'func1'
    var_19 = module_1.Constant()
    var_20 = 'func2'
    var_21 = module_1.Constant()
    var_22 = [var_19, var_21]
    var_23 = module_1.Load()
    var_24 = module_1.List()
    var_25 = 'func3'
    var_26 = module_1.Constant()
    var_27 = [var_26]
    var_28 = module_1.Load()
    var_29 = module_1.Tuple()
    var_30 = 'NoValue'
    var_31 = 'int'
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = 'a'
    var_35 = 'b'
    var_36 = module_1.Constant()
    var_37 = 'TYPED'
    var_38 = 100
    var_39 = module_1.Constant()
    var_40 = 'TypedVar'
    var_41 = 'List'
    var_42 = module_1.Load()
    var_43 = module_1.Name()
    var_44 = []
    var_45 = module_1.Load()
    var_46 = module_1.List()



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'list[int]'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'not valid python @@'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'typing.List'
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = ''
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = 'int | str'
    var_29 = module_1.Constant()
    var_30 = var_3.visit_Constant(var_29)
    var_31 = 'test_module.MyType'
    var_32 = {var_31: var_13}
    var_33 = module_0.Resolver(var_1, var_32)
    var_34 = 'MyType'
    var_35 = module_1.Constant()
    var_36 = var_33.visit_Constant(var_35)



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test is_public_family function.'
    var_1 = 'os.path.join'
    var_2 = module_0.is_public_family(var_1)
    assert var_2 is True
    var_3 = 'sys'
    var_4 = module_0.is_public_family(var_3)
    assert var_4 is True
    var_5 = 'collections.abc.Sequence'
    var_6 = module_0.is_public_family(var_5)
    assert var_6 is True
    var_7 = 'module.__init__'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = '__main__'
    var_10 = module_0.is_public_family(var_9)
    assert var_10 is True
    var_11 = 'os.__dict__'
    var_12 = module_0.is_public_family(var_11)
    assert var_12 is True
    var_13 = '_private'
    var_14 = module_0.is_public_family(var_13)
    assert var_14 is False
    var_15 = 'os._internal'
    var_16 = module_0.is_public_family(var_15)
    assert var_16 is False
    var_17 = 'module._private.func'
    var_18 = module_0.is_public_family(var_17)
    assert var_18 is False
    var_19 = 'os.path._private'
    var_20 = module_0.is_public_family(var_19)
    assert var_20 is False
    var_21 = 'os.__name__.public'
    var_22 = module_0.is_public_family(var_21)
    assert var_22 is True
    var_23 = 'public.__magic__.func'
    var_24 = module_0.is_public_family(var_23)
    assert var_24 is True
    var_25 = 'public._private.__magic__'
    var_26 = module_0.is_public_family(var_25)
    assert var_26 is False
    var_27 = ''
    var_28 = module_0.is_public_family(var_27)
    assert var_28 is True
    var_29 = 'a'
    var_30 = module_0.is_public_family(var_29)
    assert var_30 is True
    var_31 = '_'
    var_32 = module_0.is_public_family(var_31)
    assert var_32 is False
    var_33 = '__'
    var_34 = module_0.is_public_family(var_33)
    assert var_34 is True
    var_35 = 'package.module.Class.method'
    var_36 = module_0.is_public_family(var_35)
    assert var_36 is True
    var_37 = 'package._internal.Class'
    var_38 = module_0.is_public_family(var_37)
    assert var_38 is False
    var_39 = 'package.module._private_class.method'
    var_40 = module_0.is_public_family(var_39)
    assert var_40 is False



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'typing'
    var_2 = 'typing.Union'
    var_3 = {var_2: var_2}
    var_4 = module_0.Resolver(var_1, var_3)
    var_5 = 'Union[int, str]'
    var_6 = 0
    var_7 = module_1.parse(var_5)
    var_8 = var_7.body[var_6]
    var_9 = 'typing.Optional'
    var_10 = {var_9: var_9}
    var_11 = module_0.Resolver(var_1, var_10)
    var_12 = 'Optional[int]'
    var_13 = module_1.parse(var_12)
    var_14 = var_13.body[var_6]
    var_15 = 'typing.List'
    var_16 = {var_15: var_15}
    var_17 = module_0.Resolver(var_1, var_16)
    var_18 = 'List[int]'
    var_19 = module_1.parse(var_18)
    var_20 = var_19.body[var_6]
    var_21 = 'test'
    var_22 = {}
    var_23 = module_0.Resolver(var_21, var_22)
    var_24 = 'some_func()[int]'
    var_25 = module_1.parse(var_24)
    var_26 = var_25.body[var_6]
    var_27 = {var_2: var_2}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'Union[int]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_6]
    var_32 = {}
    var_33 = module_0.Resolver(var_1, var_32)
    var_34 = 'CustomType[int]'
    var_35 = module_1.parse(var_34)
    var_36 = var_35.body[var_6]



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'mymodule'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = {}
    var_9 = module_0.Resolver(var_1, var_8)
    var_10 = 'other'
    var_11 = module_1.Load()
    var_12 = 'SomeAttr'
    var_13 = module_1.Load()
    var_14 = {}
    var_15 = module_0.Resolver(var_1, var_14)
    var_16 = module_1.Load()
    var_17 = 'Dict'
    var_18 = module_1.Load()
    var_19 = 'items'
    var_20 = module_1.Load()
    var_21 = {}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = module_1.Load()
    var_24 = 'Optional'
    var_25 = module_1.Load()
    var_26 = {}
    var_27 = module_0.Resolver(var_1, var_26)
    var_28 = module_1.Load()
    var_29 = 'Union'
    var_30 = module_1.Load()
    var_31 = {}
    var_32 = module_0.Resolver(var_1, var_31)
    var_33 = module_1.Load()
    var_34 = module_1.Load()



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 'Dict'
    var_14 = module_1.Load()
    var_15 = module_1.Attribute()
    var_16 = var_3.visit_Attribute(var_15)
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = 'Optional'
    var_20 = module_1.Load()
    var_21 = module_1.Attribute()
    var_22 = var_3.visit_Attribute(var_21)
    var_23 = 'os'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = 'path'
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = var_3.visit_Attribute(var_28)
    var_30 = var_29.value
    var_31 = 'collections'
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = 'defaultdict'
    var_35 = module_1.Load()
    var_36 = module_1.Attribute()
    var_37 = var_3.visit_Attribute(var_36)
    var_38 = 42
    var_39 = module_1.Constant()
    var_40 = 'real'
    var_41 = module_1.Load()
    var_42 = module_1.Attribute()
    var_43 = var_3.visit_Attribute(var_42)
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = 'Union'
    var_47 = module_1.Load()
    var_48 = module_1.Attribute()
    var_49 = var_3.visit_Attribute(var_48)
    var_50 = module_1.Load()
    var_51 = module_1.Name()
    var_52 = 'Callable'
    var_53 = module_1.Load()
    var_54 = module_1.Attribute()
    var_55 = var_3.visit_Attribute(var_54)



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'TypeAlias'
    var_4 = 'type'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'dict[str, int]'
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'Var'
    var_11 = 'int'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = None
    var_15 = 'CONSTANT'
    var_16 = 42
    var_17 = module_1.Constant()
    var_18 = '__all__'
    var_19 = 'func1'
    var_20 = module_1.Constant()
    var_21 = 'func2'
    var_22 = module_1.Constant()
    var_23 = [var_20, var_22]
    var_24 = module_1.Load()
    var_25 = module_1.Tuple()
    var_26 = 'typed_var'
    var_27 = 'hello'
    var_28 = module_1.Constant()
    var_29 = 'str'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = module_1.Constant()
    var_33 = var_1.alias
    var_34 = len(var_33)
    var_35 = var_1.alias
    var_36 = len(var_35)
    var_37 = 'a'
    var_38 = module_1.Constant()
    var_39 = var_1.alias
    var_40 = len(var_39)
    var_41 = var_1.alias
    var_42 = len(var_41)



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_ann method.'
    var_1 = module_0.Parser()
    var_2 = []
    var_3 = 'test_module'
    var_4 = False
    var_5 = var_1.func_ann(var_3, var_2, has_self=var_4, cls_method=var_4)
    var_6 = list(var_5)
    var_7 = 'x'
    var_8 = var_1.func_ann(var_3, var_2, has_self=var_4, cls_method=var_4)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'self'
    var_12 = None
    var_13 = True
    var_14 = var_1.func_ann(var_3, var_2, has_self=var_13, cls_method=var_4)
    var_15 = list(var_14)
    var_16 = 'cls'
    var_17 = var_1.func_ann(var_3, var_2, has_self=var_13, cls_method=var_13)
    var_18 = list(var_17)
    var_19 = var_1.func_ann(var_3, var_2, has_self=var_13, cls_method=var_4)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = '*'
    var_23 = 'y'
    var_24 = var_1.func_ann(var_3, var_2, has_self=var_4, cls_method=var_4)
    var_25 = list(var_24)
    var_26 = var_1.func_ann(var_3, var_2, has_self=var_4, cls_method=var_4)
    var_27 = list(var_26)
    var_28 = 'MyClass'
    var_29 = module_1.Load()
    var_30 = module_1.Name()
    var_31 = var_1.func_ann(var_3, var_2, has_self=var_13, cls_method=var_13)
    var_32 = list(var_31)



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.test_func'
    var_4 = "def func(a: int, b: str = 'default', *args, c: float = 1.0, **kwargs) -> bool: pass"
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.args
    var_9 = var_7.returns
    var_10 = False
    var_11 = False
    var_12 = var_1.func_api(var_2, var_3, var_8, var_9, has_self=var_10, cls_method=var_11)
    var_13 = 'def method(self, x: int) -> None: pass'
    var_14 = module_1.parse(var_13)
    var_15 = var_14.body[var_11]
    var_16 = 'test_module.TestClass.method'
    var_17 = var_15.args
    var_18 = var_15.returns
    var_19 = True
    var_20 = False
    var_21 = var_1.func_api(var_2, var_16, var_17, var_18, has_self=var_19, cls_method=var_20)
    var_22 = 'test_module.TestClass.cls_method'
    var_23 = var_15.args
    var_24 = var_15.returns
    var_25 = var_1.func_api(var_2, var_22, var_23, var_24, has_self=var_19, cls_method=var_19)
    var_26 = 'def func(a, b): pass'
    var_27 = module_1.parse(var_26)
    var_28 = var_27.body[var_20]
    var_29 = 'test_module.no_ann_func'
    var_30 = var_28.args
    var_31 = None
    var_32 = False
    var_33 = False
    var_34 = var_1.func_api(var_2, var_29, var_30, var_31, has_self=var_32, cls_method=var_33)
    var_35 = 'def func(a, /, b, *args, c, **kwargs) -> int: pass'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_33]
    var_38 = 'test_module.complex_func'
    var_39 = var_37.args
    var_40 = var_37.returns
    var_41 = False
    var_42 = False
    var_43 = var_1.func_api(var_2, var_38, var_39, var_40, has_self=var_41, cls_method=var_42)



# Parsed testcases at query #22
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 'sys'
    var_8 = 'json'
    var_9 = 'j'
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 0
    var_13 = 'typing'
    var_14 = 'List'
    var_15 = 'L'
    var_16 = 'utils'
    var_17 = 'helper'
    var_18 = 1
    var_19 = 'test_module.submodule'
    var_20 = '*'
    var_21 = 'itertools'
    var_22 = 'chain'
    var_23 = 'cycle'
    var_24 = 'c'
    var_25 = 'x'
    var_26 = 'y'
    var_27 = 2
    var_28 = 'a.b.c'



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test load_docstring when module has None docstrings.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test load_docstring with empty doc dictionary.'
    var_1 = module_0.Parser()
    var_2 = 'any_root'
    var_3 = var_1.docstring
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 'Dict'
    var_14 = module_1.Load()
    var_15 = module_1.Attribute()
    var_16 = var_3.visit_Attribute(var_15)
    var_17 = 'other'
    var_18 = module_1.Load()
    var_19 = module_1.Name()
    var_20 = 'method'
    var_21 = module_1.Load()
    var_22 = module_1.Attribute()
    var_23 = var_3.visit_Attribute(var_22)
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = 'sub'
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = module_1.Load()
    var_30 = module_1.Attribute()
    var_31 = var_3.visit_Attribute(var_30)
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = 'Optional'
    var_35 = module_1.Load()
    var_36 = module_1.Attribute()
    var_37 = var_3.visit_Attribute(var_36)
    var_38 = module_1.Load()
    var_39 = module_1.Name()
    var_40 = 'Union'
    var_41 = module_1.Load()
    var_42 = module_1.Attribute()
    var_43 = var_3.visit_Attribute(var_42)



# Parsed testcases at query #25
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'typing.List'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'not valid syntax @@'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = ''
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = 'List[int]'
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = {}
    var_29 = 'T'
    var_30 = module_0.Resolver(var_1, var_28, var_29)
    var_31 = module_1.Constant()
    var_32 = var_30.visit_Constant(var_31)



# Parsed testcases at query #26
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = 'other'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Dict'
    var_13 = module_1.Load()
    var_14 = 'not_a_name'
    var_15 = 'attr'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'Optional'
    var_19 = module_1.Load()
    var_20 = 'module'
    var_21 = module_1.Load()
    var_22 = 'submodule'
    var_23 = module_1.Load()
    var_24 = module_1.Load()



# Parsed testcases at query #27
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = {}
    var_12 = module_0.Resolver(var_1, var_11)
    var_13 = 'other_module'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = 'SomeClass'
    var_17 = module_1.Load()
    var_18 = module_1.Attribute()
    var_19 = var_12.visit_Attribute(var_18)
    var_20 = var_19.value
    var_21 = {}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'obj'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = 'prop'
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = 'nested'
    var_30 = module_1.Load()
    var_31 = module_1.Attribute()
    var_32 = var_22.visit_Attribute(var_31)
    var_33 = var_32.value
    var_34 = {}
    var_35 = module_0.Resolver(var_1, var_34)
    var_36 = module_1.Load()
    var_37 = module_1.Name()
    var_38 = 'Optional'
    var_39 = module_1.Load()
    var_40 = module_1.Attribute()
    var_41 = var_35.visit_Attribute(var_40)
    var_42 = {}
    var_43 = module_0.Resolver(var_1, var_42)
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = 'Dict'
    var_47 = module_1.Load()
    var_48 = module_1.Attribute()
    var_49 = var_43.visit_Attribute(var_48)



# Parsed testcases at query #28
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring with None docstrings.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.load_docstring with empty doc dictionary.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'



# Parsed testcases at query #29
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'collections'
    var_6 = 'col'
    var_7 = 'typing'
    var_8 = 'List'
    var_9 = 0
    var_10 = ''
    var_11 = 'Dict'
    var_12 = 'D'
    var_13 = module_0.Parser()
    var_14 = 'pkg.sub.module'
    var_15 = 'utils'
    var_16 = 'helper'
    var_17 = 1
    var_18 = module_0.parent(var_14, level=var_9)
    var_19 = 'path'
    var_20 = 'getcwd'
    var_21 = 'cwd'
    var_22 = module_0.Parser()
    var_23 = 'another_module'
    var_24 = 'something'
    var_25 = module_0.Parser()
    var_26 = 'test'



# Parsed testcases at query #30
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'Test const_type function with various AST expressions.'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'int'
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'float'
    var_7 = 'hello'
    var_8 = module_0.Constant()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'str'
    var_10 = True
    var_11 = module_0.Constant()
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'bool'
    var_13 = None
    var_14 = module_0.Constant()
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'NoneType'
    var_16 = module_0.Constant()
    var_17 = 2
    var_18 = module_0.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_0.Load()
    var_21 = module_0.Tuple()
    var_22 = module_1.const_type(var_21)
    assert var_22 == 'tuple[int, int]'
    var_23 = module_0.Constant()
    var_24 = module_0.Constant()
    var_25 = [var_23, var_24]
    var_26 = module_0.Load()
    var_27 = module_0.List()
    var_28 = module_1.const_type(var_27)
    assert var_28 == 'list[int, int]'
    var_29 = module_0.Constant()
    var_30 = module_0.Constant()
    var_31 = [var_29, var_30]
    var_32 = module_0.Set()
    var_33 = module_1.const_type(var_32)
    assert var_33 == 'set[int, int]'
    var_34 = 'a'
    var_35 = module_0.Constant()
    var_36 = 'b'
    var_37 = module_0.Constant()
    var_38 = [var_35, var_37]
    var_39 = module_0.Constant()
    var_40 = module_0.Constant()
    var_41 = [var_39, var_40]
    var_42 = module_0.Dict()
    var_43 = module_1.const_type(var_42)
    assert var_43 == 'dict[str, int]'
    var_44 = module_0.Constant()
    var_45 = 'str'
    var_46 = module_0.Constant()
    var_47 = [var_44, var_46]
    var_48 = module_0.Load()
    var_49 = module_0.Tuple()
    var_50 = module_1.const_type(var_49)
    var_51 = []
    var_52 = module_0.Load()
    var_53 = module_0.Tuple()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'tuple'
    var_55 = 'bool'
    var_56 = module_0.Load()
    var_57 = module_0.Name()
    var_58 = []
    var_59 = []
    var_60 = module_0.Call(*var_58)
    var_61 = module_1.const_type(var_60)
    assert var_61 == 'bool'
    var_62 = 'int'
    var_63 = module_0.Load()
    var_64 = module_0.Name()
    var_65 = []
    var_66 = []
    var_67 = module_0.Call(*var_65)
    var_68 = module_1.const_type(var_67)
    assert var_68 == 'int'
    var_69 = 'float'
    var_70 = module_0.Load()
    var_71 = module_0.Name()
    var_72 = []
    var_73 = []
    var_74 = module_0.Call(*var_72)
    var_75 = module_1.const_type(var_74)
    assert var_75 == 'float'
    var_76 = module_0.Load()
    var_77 = module_0.Name()
    var_78 = []
    var_79 = []
    var_80 = module_0.Call(*var_78)
    var_81 = module_1.const_type(var_80)
    assert var_81 == 'str'
    var_82 = 'complex'
    var_83 = module_0.Load()
    var_84 = module_0.Name()
    var_85 = []
    var_86 = []
    var_87 = module_0.Call(*var_85)
    var_88 = module_1.const_type(var_87)
    assert var_88 == 'complex'
    var_89 = 'unknown_func'
    var_90 = module_0.Load()
    var_91 = module_0.Name()
    var_92 = []
    var_93 = []
    var_94 = module_0.Call(*var_92)
    var_95 = module_1.const_type(var_94)
    var_96 = 'x'
    var_97 = module_0.Load()
    var_98 = module_0.Name()
    var_99 = module_1.const_type(var_98)
    var_100 = module_0.Constant()
    var_101 = module_0.BitOr()
    var_102 = module_0.Constant()
    var_103 = module_0.BinOp()
    var_104 = module_1.const_type(var_103)
    var_105 = module_0.Constant()
    var_106 = [var_13, var_105]
    var_107 = module_0.Load()
    var_108 = module_0.Tuple()
    var_109 = module_1.const_type(var_108)
    var_110 = module_0.Constant()
    var_111 = [var_110]
    var_112 = module_0.Constant()
    var_113 = [var_112]
    var_114 = module_0.Dict()
    var_115 = module_1.const_type(var_114)
    assert var_115 == 'dict[str, int]'
    var_116 = module_0.Constant()
    var_117 = module_0.Constant()
    var_118 = [var_116, var_117]
    var_119 = module_0.Load()
    var_120 = module_0.List()
    var_121 = module_1.const_type(var_120)



# Parsed testcases at query #31
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'typing.List'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = '@#$%^&*()'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'not a valid expression !!!'
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = ''
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = 'test_module.MyType'
    var_29 = 'typing.List[int]'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_1, var_30)
    var_32 = 'MyType'
    var_33 = module_1.Constant()
    var_34 = var_31.visit_Constant(var_33)
    var_35 = {}
    var_36 = 'T'
    var_37 = module_0.Resolver(var_1, var_35, var_36)
    var_38 = module_1.Constant()
    var_39 = var_37.visit_Constant(var_38)



# Parsed testcases at query #32
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = {}
    var_12 = module_0.Resolver(var_1, var_11)
    var_13 = 'other_module'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = 'SomeClass'
    var_17 = module_1.Load()
    var_18 = module_1.Attribute()
    var_19 = var_12.visit_Attribute(var_18)
    var_20 = {}
    var_21 = module_0.Resolver(var_1, var_20)
    var_22 = 'obj'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = 'prop'
    var_26 = module_1.Load()
    var_27 = module_1.Attribute()
    var_28 = 'nested'
    var_29 = module_1.Load()
    var_30 = module_1.Attribute()
    var_31 = var_21.visit_Attribute(var_30)
    var_32 = {}
    var_33 = module_0.Resolver(var_1, var_32)
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = 'Dict'
    var_37 = module_1.Load()
    var_38 = module_1.Attribute()
    var_39 = var_33.visit_Attribute(var_38)
    var_40 = {}
    var_41 = module_0.Resolver(var_1, var_40)
    var_42 = module_1.Load()
    var_43 = module_1.Name()
    var_44 = 'Union'
    var_45 = module_1.Load()
    var_46 = module_1.Attribute()
    var_47 = var_41.visit_Attribute(var_46)



# Parsed testcases at query #33
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'List[str]'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'not valid python @@@@'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = ''
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = '  invalid  '
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = 'test_module.MyType'
    var_29 = 'List[int]'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_1, var_30)
    var_32 = 'MyType'
    var_33 = module_1.Constant()
    var_34 = var_31.visit_Constant(var_33)



# Parsed testcases at query #34
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'MyType'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'typing.List'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'not a valid python expression !!!'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = '   '
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = ''
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = 'List[int]'
    var_29 = module_1.Constant()
    var_30 = var_3.visit_Constant(var_29)
    var_31 = 'test_module.MyType'
    var_32 = 'str'
    var_33 = {var_31: var_32}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = module_1.Constant()
    var_36 = var_34.visit_Constant(var_35)



# Parsed testcases at query #35
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = "\nclass MyClass(BaseClass, OtherBase):\n    '''Class docstring'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = module_1.parse(var_2)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = 'test_module.MyClass'
    var_9 = var_7.bases
    var_10 = var_7.body
    var_11 = var_1.class_api(var_3, var_8, var_9, var_10)
    var_12 = "\nclass MyClass2:\n    '''Class docstring'''\n    attr1: int = 5\n    attr2: str\n    _private: float = 3.14\n"
    var_13 = 'test_module2'
    var_14 = module_0.Parser()
    var_15 = var_14.parse(var_13, var_12)
    var_16 = module_1.parse(var_12)
    var_17 = var_16.body[var_6]
    var_18 = 'test_module2.MyClass2'
    var_19 = var_17.bases
    var_20 = var_17.body
    var_21 = var_14.class_api(var_13, var_18, var_19, var_20)
    var_22 = '\nfrom enum import Enum\n\nclass MyEnum(Enum):\n    \'\'\'Enum docstring\'\'\'\n    VALUE1: int = 1\n    VALUE2: str = "test"\n'
    var_23 = 'test_module3'
    var_24 = module_0.Parser()
    var_25 = var_24.parse(var_23, var_22)
    var_26 = module_1.parse(var_22)
    var_27 = 1
    var_28 = var_26.body[var_27]
    var_29 = 'test_module3.MyEnum'
    var_30 = var_28.bases
    var_31 = var_28.body
    var_32 = var_24.class_api(var_23, var_29, var_30, var_31)
    var_33 = '\nclass MyClass4:\n    \'\'\'Class docstring\'\'\'\n    attr1: int = 1\n    attr2: str = "test"\n    del attr1\n'
    var_34 = 'test_module4'
    var_35 = module_0.Parser()
    var_36 = var_35.parse(var_34, var_33)
    var_37 = module_1.parse(var_33)
    var_38 = var_37.body[var_6]
    var_39 = 'test_module4.MyClass4'
    var_40 = var_38.bases
    var_41 = var_38.body
    var_42 = var_35.class_api(var_34, var_39, var_40, var_41)
    var_43 = '\nclass EmptyClass:\n    pass\n'
    var_44 = 'test_module5'
    var_45 = module_0.Parser()
    var_46 = var_45.parse(var_44, var_43)
    var_47 = module_1.parse(var_43)
    var_48 = var_47.body[var_6]
    var_49 = 'test_module5.EmptyClass'
    var_50 = var_48.bases
    var_51 = var_48.body
    var_52 = var_45.class_api(var_44, var_49, var_50, var_51)



# Parsed testcases at query #36
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass'
    var_4 = '\nclass MyClass(BaseClass, AnotherBase):\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.bases
    var_9 = var_7.body
    var_10 = var_1.class_api(var_2, var_3, var_8, var_9)
    var_11 = '\nclass MyClass:\n    public_attr: int\n    _private_attr: str\n    another_member: float = 3.14\n'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_6]
    var_14 = var_13.bases
    var_15 = var_13.body
    var_16 = var_1.class_api(var_2, var_3, var_14, var_15)
    var_17 = '\nclass MyEnum(enum.Enum):\n    OPTION_A: int\n    OPTION_B: str\n'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_6]
    var_20 = var_19.bases
    var_21 = var_19.body
    var_22 = var_1.class_api(var_2, var_3, var_20, var_21)
    var_23 = 'class EmptyClass: pass'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_6]
    var_26 = []
    var_27 = var_25.body
    var_28 = var_1.class_api(var_2, var_3, var_26, var_27)
    var_29 = '\nclass MyClass:\n    attr: int\n    del attr\n'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_6]
    var_32 = var_31.bases
    var_33 = var_31.body
    var_34 = var_1.class_api(var_2, var_3, var_32, var_33)
    var_35 = '\nclass MyClass:\n    value = 42  # type: int\n'
    var_36 = True
    var_37 = module_1.parse(var_35, type_comments=var_36)
    var_38 = var_37.body[var_6]
    var_39 = var_38.bases
    var_40 = var_38.body
    var_41 = var_1.class_api(var_2, var_3, var_39, var_40)



# Parsed testcases at query #37
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'mymodule'
    var_2 = 'mymodule.Union'
    var_3 = 'typing.Union'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 'Union[int, str]'
    var_7 = 0
    var_8 = module_1.parse(var_6)
    var_9 = var_8.body[var_7]
    var_10 = var_9.value
    var_11 = 'mymodule.Optional'
    var_12 = 'typing.Optional'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'Optional[int]'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_7]
    var_18 = var_17.value
    var_19 = {}
    var_20 = module_0.Resolver(var_1, var_19)
    var_21 = 'obj.Optional[int]'
    var_22 = module_1.parse(var_21)
    var_23 = var_22.body[var_7]
    var_24 = var_23.value
    var_25 = 'mymodule.Dict'
    var_26 = 'typing.Dict'
    var_27 = {var_25: var_26}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'Dict[str, int]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_7]
    var_32 = var_31.value
    var_33 = {var_2: var_3}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = 'Union[int]'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_7]
    var_38 = var_37.value
    var_39 = {}
    var_40 = module_0.Resolver(var_1, var_39)
    var_41 = 'CustomType[int]'
    var_42 = module_1.parse(var_41)
    var_43 = var_42.body[var_7]
    var_44 = var_43.value
    var_45 = {var_2: var_3}
    var_46 = module_0.Resolver(var_1, var_45)
    var_47 = 'Union[int, str, float]'
    var_48 = module_1.parse(var_47)
    var_49 = var_48.body[var_7]
    var_50 = var_49.value



# Parsed testcases at query #38
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Load()
    var_9 = 'Dict'
    var_10 = module_1.Load()
    var_11 = 'other'
    var_12 = module_1.Load()
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = 'Optional'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'collections'
    var_19 = module_1.Load()
    var_20 = 'abc'
    var_21 = module_1.Load()



# Parsed testcases at query #39
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 'os.path'
    var_8 = 'join'
    var_9 = 0
    var_10 = ''
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 'dd'
    var_14 = 'package.submodule.test'
    var_15 = 'utils'
    var_16 = 'helper'
    var_17 = 1
    var_18 = module_0.parent(var_14, level=var_9)
    var_19 = 'config'
    var_20 = 'settings'
    var_21 = 2
    var_22 = module_0.parent(var_14, level=var_17)
    var_23 = 'typing'
    var_24 = 'List'
    var_25 = 'Dict'
    var_26 = 'something'
    var_27 = module_0.parent(var_2, level=var_9)



# Parsed testcases at query #40
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'invalid syntax @@'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'List[str]'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'test_module.MyType'
    var_23 = {var_22: var_13}
    var_24 = module_0.Resolver(var_1, var_23)
    var_25 = 'MyType'
    var_26 = module_1.Constant()
    var_27 = var_24.visit_Constant(var_26)
    var_28 = ''
    var_29 = module_1.Constant()
    var_30 = var_3.visit_Constant(var_29)



# Parsed testcases at query #41
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Name method.'
    var_1 = 'mymodule'
    var_2 = {}
    var_3 = 'MyClass'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = module_1.Load()
    var_6 = {}
    var_7 = ''
    var_8 = module_0.Resolver(var_1, var_6, var_7)
    var_9 = 'SomeName'
    var_10 = module_1.Load()
    var_11 = 'mymodule.MyType'
    var_12 = 'int'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'MyType'
    var_16 = module_1.Load()
    var_17 = 'mymodule.T'
    var_18 = 'typing.TypeVar'
    var_19 = "typing.TypeVar('T')"
    var_20 = {var_17: var_19, var_18: var_18}
    var_21 = module_0.Resolver(var_1, var_20)
    var_22 = 'T'
    var_23 = module_1.Load()
    var_24 = {var_11: var_11}
    var_25 = module_0.Resolver(var_1, var_24)
    var_26 = module_1.Load()
    var_27 = 'mymodule.Container'
    var_28 = 'list[int]'
    var_29 = {var_27: var_28}
    var_30 = module_0.Resolver(var_1, var_29)
    var_31 = 'Container'
    var_32 = module_1.Load()
    var_33 = 'other.Type'
    var_34 = 'str'
    var_35 = {var_33: var_34}
    var_36 = module_0.Resolver(var_1, var_35)
    var_37 = 'Type'
    var_38 = module_1.Load()



# Parsed testcases at query #42
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'var1'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'var2'
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = None
    var_15 = 'var3'
    var_16 = 3.14
    var_17 = module_1.Constant()
    var_18 = 'float'
    var_19 = 'var4'
    var_20 = 'hello'
    var_21 = module_1.Constant()
    var_22 = 'CONST'
    var_23 = 100
    var_24 = module_1.Constant()
    var_25 = '__all__'
    var_26 = 'func1'
    var_27 = module_1.Constant()
    var_28 = 'Class1'
    var_29 = module_1.Constant()
    var_30 = [var_27, var_29]
    var_31 = module_1.Load()
    var_32 = module_1.Tuple()
    var_33 = 'a'
    var_34 = module_1.Constant()
    var_35 = 'x'
    var_36 = 'y'
    var_37 = module_1.Constant()



# Parsed testcases at query #43
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'MyType'
    var_4 = 'list'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.Load()
    var_11 = module_1.Subscript()
    var_12 = []
    var_13 = module_1.Load()
    var_14 = module_1.List()
    var_15 = 1
    var_16 = set()
    var_17 = 'CONSTANT'
    var_18 = 'int'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = 42
    var_22 = module_1.Constant()
    var_23 = 'VAR'
    var_24 = 'string'
    var_25 = module_1.Constant()
    var_26 = '__all__'
    var_27 = 'func'
    var_28 = module_1.Constant()
    var_29 = 'Class'
    var_30 = module_1.Constant()
    var_31 = [var_28, var_30]
    var_32 = module_1.Load()
    var_33 = module_1.Tuple()
    var_34 = 'TYPED'
    var_35 = 3.14
    var_36 = module_1.Constant()
    var_37 = 'float'
    var_38 = 'x'
    var_39 = 'y'
    var_40 = module_1.Constant()



# Parsed testcases at query #44
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'Test const_type function with various AST nodes.'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'int'
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'float'
    var_7 = 'hello'
    var_8 = module_0.Constant()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'str'
    var_10 = True
    var_11 = module_0.Constant()
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'bool'
    var_13 = None
    var_14 = module_0.Constant()
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'NoneType'
    var_16 = module_0.Constant()
    var_17 = 2
    var_18 = module_0.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_0.Load()
    var_21 = module_0.Tuple()
    var_22 = module_1.const_type(var_21)
    assert var_22 == 'tuple[int, int]'
    var_23 = module_0.Constant()
    var_24 = 'a'
    var_25 = module_0.Constant()
    var_26 = [var_23, var_25]
    var_27 = module_0.Load()
    var_28 = module_0.Tuple()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'tuple[Any, Any]'
    var_30 = module_0.Constant()
    var_31 = module_0.Constant()
    var_32 = [var_30, var_31]
    var_33 = module_0.Load()
    var_34 = module_0.List()
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'list[int, int]'
    var_36 = []
    var_37 = module_0.Load()
    var_38 = module_0.List()
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'list'
    var_40 = module_0.Constant()
    var_41 = module_0.Constant()
    var_42 = [var_40, var_41]
    var_43 = module_0.Set()
    var_44 = module_1.const_type(var_43)
    assert var_44 == 'set[int, int]'
    var_45 = module_0.Constant()
    var_46 = [var_45]
    var_47 = module_0.Constant()
    var_48 = [var_47]
    var_49 = module_0.Dict()
    var_50 = module_1.const_type(var_49)
    assert var_50 == 'dict[str, int]'
    var_51 = []
    var_52 = []
    var_53 = module_0.Dict()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'dict'
    var_55 = 'int'
    var_56 = module_0.Load()
    var_57 = module_0.Name()
    var_58 = []
    var_59 = []
    var_60 = module_0.Call(*var_58)
    var_61 = module_1.const_type(var_60)
    assert var_61 == 'int'
    var_62 = 'str'
    var_63 = module_0.Load()
    var_64 = module_0.Name()
    var_65 = []
    var_66 = []
    var_67 = module_0.Call(*var_65)
    var_68 = module_1.const_type(var_67)
    assert var_68 == 'str'
    var_69 = 'bool'
    var_70 = module_0.Load()
    var_71 = module_0.Name()
    var_72 = []
    var_73 = []
    var_74 = module_0.Call(*var_72)
    var_75 = module_1.const_type(var_74)
    assert var_75 == 'bool'
    var_76 = 'float'
    var_77 = module_0.Load()
    var_78 = module_0.Name()
    var_79 = []
    var_80 = []
    var_81 = module_0.Call(*var_79)
    var_82 = module_1.const_type(var_81)
    assert var_82 == 'float'
    var_83 = 'complex'
    var_84 = module_0.Load()
    var_85 = module_0.Name()
    var_86 = []
    var_87 = []
    var_88 = module_0.Call(*var_86)
    var_89 = module_1.const_type(var_88)
    assert var_89 == 'complex'
    var_90 = 'module'
    var_91 = module_0.Load()
    var_92 = module_0.Name()
    var_93 = 'list'
    var_94 = module_0.Load()
    var_95 = module_0.Attribute()
    var_96 = []
    var_97 = []
    var_98 = module_0.Call(*var_96)
    var_99 = module_1.const_type(var_98)
    var_100 = 'unknown'
    var_101 = module_0.Load()
    var_102 = module_0.Name()
    var_103 = []
    var_104 = []
    var_105 = module_0.Call(*var_103)
    var_106 = module_1.const_type(var_105)
    var_107 = module_0.Constant()
    var_108 = module_0.BitOr()
    var_109 = module_0.Constant()
    var_110 = module_0.BinOp()
    var_111 = module_1.const_type(var_110)
    var_112 = 'x'
    var_113 = module_0.Load()
    var_114 = module_0.Name()
    var_115 = module_1.const_type(var_114)
    var_116 = module_0.Constant()
    var_117 = [var_116, var_13]
    var_118 = module_0.Load()
    var_119 = module_0.Tuple()
    var_120 = module_1.const_type(var_119)
    assert var_120 == 'tuple'
    var_121 = module_0.Constant()
    var_122 = module_0.Constant()
    var_123 = [var_121, var_122]
    var_124 = module_0.Load()
    var_125 = module_0.List()
    var_126 = module_1.const_type(var_125)
    assert var_126 == 'list[Any, Any]'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test is_public_family function.'
    var_1 = 'module'
    var_2 = module_0.is_public_family(var_1)
    var_3 = 'module.submodule'
    var_4 = module_0.is_public_family(var_3)
    var_5 = 'module.submodule.Class'
    var_6 = module_0.is_public_family(var_5)
    var_7 = 'module.Class.method'
    var_8 = module_0.is_public_family(var_7)
    var_9 = 'module.__init__'
    var_10 = module_0.is_public_family(var_9)
    var_11 = 'module.__init__.submodule'
    var_12 = module_0.is_public_family(var_11)
    var_13 = '__main__'
    var_14 = module_0.is_public_family(var_13)
    var_15 = '__main__.module'
    var_16 = module_0.is_public_family(var_15)
    var_17 = 'module.__dict__.submodule'
    var_18 = module_0.is_public_family(var_17)
    var_19 = '_private'
    var_20 = module_0.is_public_family(var_19)
    var_21 = 'module._private'
    var_22 = module_0.is_public_family(var_21)
    var_23 = 'module._private.Class'
    var_24 = module_0.is_public_family(var_23)
    var_25 = 'module.Class._private'
    var_26 = module_0.is_public_family(var_25)
    var_27 = '_module.public'
    var_28 = module_0.is_public_family(var_27)
    var_29 = 'module.normal_name'
    var_30 = module_0.is_public_family(var_29)
    var_31 = 'module.name_with_underscores'
    var_32 = module_0.is_public_family(var_31)
    var_33 = 'A'
    var_34 = module_0.is_public_family(var_33)
    var_35 = '_A'
    var_36 = module_0.is_public_family(var_35)
    var_37 = '__A__'
    var_38 = module_0.is_public_family(var_37)
    var_39 = 'module.__A'
    var_40 = module_0.is_public_family(var_39)
    var_41 = 'module.A_'
    var_42 = module_0.is_public_family(var_41)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'CONST'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 100
    var_14 = module_1.Constant()
    var_15 = 'y'
    var_16 = 'hello'
    var_17 = module_1.Constant()
    var_18 = None
    var_19 = 'z'
    var_20 = 3.14
    var_21 = module_1.Constant()
    var_22 = 'float'
    var_23 = '__all__'
    var_24 = 'list'
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = 'public_func'
    var_28 = module_1.Constant()
    var_29 = [var_28]
    var_30 = module_1.Load()
    var_31 = module_1.List()
    var_32 = 'func1'
    var_33 = module_1.Constant()
    var_34 = 'func2'
    var_35 = module_1.Constant()
    var_36 = [var_33, var_35]
    var_37 = module_1.Load()
    var_38 = module_1.Tuple()
    var_39 = 'a'
    var_40 = 'b'
    var_41 = module_1.Constant()
    var_42 = var_1.alias
    var_43 = len(var_42)
    var_44 = var_1.alias
    var_45 = len(var_44)
    var_46 = 'unassigned'
    var_47 = 'str'
    var_48 = module_1.Load()
    var_49 = module_1.Name()
    var_50 = var_1.alias
    var_51 = len(var_50)
    var_52 = var_1.alias
    var_53 = len(var_52)
    var_54 = 'local_var'
    var_55 = module_1.Constant()



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.compile method.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    var_3 = '\n'
    var_4 = module_0.Parser()
    var_5 = 'test_module'
    var_6 = 'def foo(): pass'
    var_7 = var_4.parse(var_5, var_6)
    var_8 = var_4.compile()
    var_9 = True
    var_10 = module_0.Parser(toc=var_9)
    var_11 = var_10.parse(var_5, var_6)
    var_12 = var_10.compile()
    var_13 = module_0.Parser()
    var_14 = 'pkg.module1'
    var_15 = 'def func1(): pass'
    var_16 = var_13.parse(var_14, var_15)
    var_17 = 'pkg.module2'
    var_18 = 'def func2(): pass'
    var_19 = var_13.parse(var_17, var_18)
    var_20 = var_13.compile()
    var_21 = module_0.Parser()
    var_22 = 'CONST = 42'
    var_23 = var_21.parse(var_5, var_22)
    var_24 = var_21.compile()
    var_25 = module_0.Parser()
    var_26 = var_25.compile()
    var_27 = module_0.Parser(var_9)
    var_28 = var_27.compile()
    var_29 = module_0.Parser()
    var_30 = var_29.compile()
    var_31 = module_0.Parser()
    var_32 = '__all__ = ["func1"]'
    var_33 = var_31.parse(var_5, var_32)
    var_34 = var_31.compile()
    var_35 = module_0.Parser()
    var_36 = var_35.compile()



# Parsed testcases at query #4
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = "\ndef example_func(x: int) -> str:\n    '''Example function.'''\n    return str(x)\n"
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_1.api(var_2, var_6)
    var_8 = var_6.name
    var_9 = "\nasync def async_func() -> None:\n    '''Async function.'''\n    pass\n"
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body[var_5]
    var_12 = var_1.api(var_2, var_11)
    var_13 = var_11.name
    var_14 = "\nclass ExampleClass:\n    '''Example class.'''\n    pass\n"
    var_15 = module_1.parse(var_14)
    var_16 = var_15.body[var_5]
    var_17 = var_1.api(var_2, var_16)
    var_18 = var_16.name
    var_19 = "\n@staticmethod\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_20 = module_1.parse(var_19)
    var_21 = var_20.body[var_5]
    var_22 = var_1.api(var_2, var_21)
    var_23 = var_21.name
    var_24 = 'OuterClass'
    var_25 = var_1.api(var_2, var_6, prefix=var_24)
    var_26 = var_6.name



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
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
    var_19 = var_18.op
    var_20 = {}
    var_21 = module_0.Resolver(var_1, var_20)
    var_22 = 'Optional'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = module_1.Load()
    var_28 = module_1.Subscript()
    var_29 = var_21.visit_Subscript(var_28)
    var_30 = var_29.op
    var_31 = var_29.right
    var_32 = {}
    var_33 = module_0.Resolver(var_1, var_32)
    var_34 = 'List'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = module_1.Load()
    var_40 = module_1.Subscript()
    var_41 = var_33.visit_Subscript(var_40)
    var_42 = var_41.value
    var_43 = {}
    var_44 = module_0.Resolver(var_1, var_43)
    var_45 = 'typing'
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = 'Dict'
    var_49 = module_1.Load()
    var_50 = module_1.Attribute()
    var_51 = module_1.Load()
    var_52 = module_1.Name()
    var_53 = module_1.Load()
    var_54 = module_1.Subscript()
    var_55 = var_44.visit_Subscript(var_54)
    var_56 = {}
    var_57 = module_0.Resolver(var_1, var_56)
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = module_1.Load()
    var_61 = module_1.Name()
    var_62 = module_1.Load()
    var_63 = module_1.Subscript()
    var_64 = var_57.visit_Subscript(var_63)
    var_65 = 'test_module.Union'
    var_66 = 'typing.Union'
    var_67 = {var_65: var_66}
    var_68 = module_0.Resolver(var_1, var_67)
    var_69 = module_1.Load()
    var_70 = module_1.Name()
    var_71 = module_1.Load()
    var_72 = module_1.Name()
    var_73 = module_1.Load()
    var_74 = module_1.Name()
    var_75 = [var_72, var_74]
    var_76 = module_1.Load()
    var_77 = module_1.Tuple()
    var_78 = module_1.Load()
    var_79 = module_1.Subscript()
    var_80 = var_68.visit_Subscript(var_79)
    var_81 = var_80.op
    var_82 = {}
    var_83 = module_0.Resolver(var_1, var_82)
    var_84 = 'CustomType'
    var_85 = module_1.Load()
    var_86 = module_1.Name()
    var_87 = module_1.Load()
    var_88 = module_1.Name()
    var_89 = module_1.Load()
    var_90 = module_1.Subscript()
    var_91 = var_83.visit_Subscript(var_90)



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.func_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.test_func'
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = module_1.arguments(*var_5)
    var_11 = None
    var_12 = False
    var_13 = var_1.func_api(var_2, var_3, var_10, var_11, has_self=var_12, cls_method=var_12)
    var_14 = 'test_module2'
    var_15 = 'test_module2.func_with_args'
    var_16 = 'x'
    var_17 = 'int'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_12]
    var_20 = var_19.value
    var_21 = module_1.arg()
    var_22 = 'y'
    var_23 = 'str'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_12]
    var_26 = var_25.value
    var_27 = module_1.arg()
    var_28 = '""'
    var_29 = module_1.parse(var_28)
    var_30 = var_29.body[var_12]
    var_31 = var_30.value
    var_32 = []
    var_33 = [var_21, var_27]
    var_34 = []
    var_35 = []
    var_36 = [var_31]
    var_37 = module_1.arguments(*var_33)
    var_38 = var_1.func_api(var_14, var_15, var_37, var_6, has_self=var_12, cls_method=var_12)
    var_39 = 'test_module3'
    var_40 = 'test_module3.MyClass.method'
    var_41 = 'self'
    var_42 = module_1.arg()
    var_43 = 'value'
    var_44 = module_1.parse(var_17)
    var_45 = var_44.body[var_12]
    var_46 = var_45.value
    var_47 = module_1.arg()
    var_48 = []
    var_49 = [var_42, var_47]
    var_50 = []
    var_51 = []
    var_52 = []
    var_53 = module_1.arguments(*var_49)
    var_54 = True
    var_55 = var_1.func_api(var_39, var_40, var_53, var_6, has_self=var_54, cls_method=var_12)
    var_56 = 'test_module4'
    var_57 = 'test_module4.MyClass.cls_method'
    var_58 = 'cls'
    var_59 = 'type[MyClass]'
    var_60 = module_1.parse(var_59)
    var_61 = var_60.body[var_12]
    var_62 = var_61.value
    var_63 = module_1.arg()
    var_64 = []
    var_65 = [var_63]
    var_66 = []
    var_67 = []
    var_68 = []
    var_69 = module_1.arguments(*var_65)
    var_70 = var_1.func_api(var_56, var_57, var_69, var_6, has_self=var_54, cls_method=var_54)
    var_71 = 'test_module5'
    var_72 = 'test_module5.var_func'
    var_73 = 'args'
    var_74 = module_1.arg()
    var_75 = 'kwargs'
    var_76 = module_1.arg()
    var_77 = []
    var_78 = []
    var_79 = []
    var_80 = []
    var_81 = []
    var_82 = module_1.arguments(*var_78)
    var_83 = var_1.func_api(var_71, var_72, var_82, var_6, has_self=var_12, cls_method=var_12)
    var_84 = 'test_module6'
    var_85 = 'test_module6.kw_func'
    var_86 = 'kwonly'
    var_87 = 'bool'
    var_88 = module_1.parse(var_87)
    var_89 = var_88.body[var_12]
    var_90 = var_89.value
    var_91 = module_1.arg()
    var_92 = []
    var_93 = []
    var_94 = [var_91]
    var_95 = [var_6]
    var_96 = []
    var_97 = module_1.arguments(*var_93)
    var_98 = var_1.func_api(var_84, var_85, var_97, var_6, has_self=var_12, cls_method=var_12)



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 'x'
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = 'int'
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = '@invalid syntax!'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 3.14
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = None
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'List[int]'
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = 'typing.List'
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 'collections'
    var_8 = 'defaultdict'
    var_9 = 0
    var_10 = 'typing'
    var_11 = 'List'
    var_12 = 'L'
    var_13 = 'sibling'
    var_14 = 'func'
    var_15 = 1
    var_16 = 'parent.test_module'
    var_17 = 'module'
    var_18 = 'item'
    var_19 = 2
    var_20 = 'parent.child.test_module'
    var_21 = 'math'
    var_22 = 'sin'
    var_23 = 'cos'
    var_24 = 'cosine'
    var_25 = 'pi'
    var_26 = 'local_func'
    var_27 = 'package.module'



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.globals method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'MyType'
    var_4 = 'str'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'SomeType'
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'CONSTANT'
    var_11 = 'int'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 42
    var_15 = module_1.Constant()
    var_16 = 'var'
    var_17 = 100
    var_18 = module_1.Constant()
    var_19 = 'num'
    var_20 = 3.14
    var_21 = module_1.Constant()
    var_22 = None
    var_23 = '__all__'
    var_24 = 'func1'
    var_25 = module_1.Constant()
    var_26 = 'Class1'
    var_27 = module_1.Constant()
    var_28 = [var_25, var_27]
    var_29 = module_1.Load()
    var_30 = module_1.List()
    var_31 = 'item1'
    var_32 = module_1.Constant()
    var_33 = [var_32]
    var_34 = module_1.Load()
    var_35 = module_1.Tuple()
    var_36 = 'a'
    var_37 = 'b'
    var_38 = module_1.Constant()
    var_39 = var_1.alias
    var_40 = len(var_39)
    var_41 = var_1.alias
    var_42 = len(var_41)
    var_43 = 'annotated'
    var_44 = module_1.Load()
    var_45 = module_1.Name()
    var_46 = var_1.alias
    var_47 = len(var_46)
    var_48 = var_1.alias
    var_49 = len(var_48)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.globals method for handling assignments and annotations.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'TypeAlias'
    var_4 = 'str'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'test'
    var_8 = module_1.Constant()
    var_9 = 1
    var_10 = 'CONST'
    var_11 = 42
    var_12 = module_1.Constant()
    var_13 = None
    var_14 = 'VAR'
    var_15 = 3.14
    var_16 = module_1.Constant()
    var_17 = 'float'
    var_18 = '__all__'
    var_19 = 'func1'
    var_20 = module_1.Constant()
    var_21 = 'Class1'
    var_22 = module_1.Constant()
    var_23 = [var_20, var_22]
    var_24 = module_1.Load()
    var_25 = module_1.Tuple()
    var_26 = 'x'
    var_27 = 'y'
    var_28 = module_1.Constant()
    var_29 = var_1.alias
    var_30 = len(var_29)
    var_31 = var_1.alias
    var_32 = len(var_31)
    var_33 = 'int'
    var_34 = module_1.Load()
    var_35 = module_1.Name()



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test_module'
    var_2 = 'test_module.Union'
    var_3 = 'typing.Union'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 'Union[int, str]'
    var_7 = 0
    var_8 = module_1.parse(var_6)
    var_9 = var_8.body[var_7]
    var_10 = var_9.value
    var_11 = 'test_module.Optional'
    var_12 = 'typing.Optional'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'Optional[int]'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_7]
    var_18 = var_17.value
    var_19 = 'test_module.List'
    var_20 = 'typing.List'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'List[int]'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_7]
    var_26 = var_25.value
    var_27 = {}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'some_attr.Union[int, str]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_7]
    var_32 = var_31.value
    var_33 = {}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = 'SomeType[int]'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_7]
    var_38 = var_37.value
    var_39 = {var_2: var_3}
    var_40 = module_0.Resolver(var_1, var_39)
    var_41 = 'Union[int, str, float]'
    var_42 = module_1.parse(var_41)
    var_43 = var_42.body[var_7]
    var_44 = var_43.value
    var_45 = {var_11: var_12}
    var_46 = module_0.Resolver(var_1, var_45)
    var_47 = 'Optional[int]'
    var_48 = module_1.parse(var_47)
    var_49 = var_48.body[var_7]
    var_50 = var_49.value



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 'typing'
    var_8 = 'List'
    var_9 = 0
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'dd'
    var_13 = 'submodule'
    var_14 = 'func'
    var_15 = 1
    var_16 = module_0.parent(var_2, level=var_9)
    var_17 = 'os.path'
    var_18 = 'join'
    var_19 = 'exists'
    var_20 = 'path_exists'
    var_21 = 'utils'
    var_22 = 'helper'
    var_23 = 2
    var_24 = module_0.parent(var_2, level=var_15)



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Unit test for Parser.func_ann method.'
    var_1 = module_0.Parser()
    var_2 = 'def method(self, x: int, y: str) -> bool: pass'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_5.args
    var_8 = True
    var_9 = False
    var_10 = var_1.func_ann(var_6, var_7, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)
    var_12 = var_5.args
    var_13 = var_1.func_ann(var_6, var_12, has_self=var_8, cls_method=var_8)
    var_14 = list(var_13)
    var_15 = 'def func(a, b, c): pass'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_9]
    var_18 = var_17.args
    var_19 = False
    var_20 = False
    var_21 = var_1.func_ann(var_6, var_18, has_self=var_19, cls_method=var_20)
    var_22 = list(var_21)
    var_23 = 'Any'
    var_24 = 'def func(a: int, *args: str, **kwargs: float) -> None: pass'
    var_25 = module_1.parse(var_24)
    var_26 = var_25.body[var_20]
    var_27 = var_26.args
    var_28 = False
    var_29 = False
    var_30 = var_1.func_ann(var_6, var_27, has_self=var_28, cls_method=var_29)
    var_31 = list(var_30)
    var_32 = 'def func(a: int, *, b: str) -> bool: pass'
    var_33 = module_1.parse(var_32)
    var_34 = var_33.body[var_29]
    var_35 = var_34.args
    var_36 = False
    var_37 = False
    var_38 = var_1.func_ann(var_6, var_35, has_self=var_36, cls_method=var_37)
    var_39 = list(var_38)
    var_40 = 'def func(a: int, /, b: str) -> bool: pass'
    var_41 = module_1.parse(var_40)
    var_42 = var_41.body[var_37]
    var_43 = var_42.args
    var_44 = False
    var_45 = False
    var_46 = var_1.func_ann(var_6, var_43, has_self=var_44, cls_method=var_45)
    var_47 = list(var_46)
    var_48 = 'test_module.MyClass'
    var_49 = 'MyClass'
    var_50 = "def method(self: 'MyClass', x: int) -> None: pass"
    var_51 = module_1.parse(var_50)
    var_52 = var_51.body[var_45]
    var_53 = var_52.args
    var_54 = var_1.func_ann(var_6, var_53, has_self=var_8, cls_method=var_8)
    var_55 = list(var_54)



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Load()
    var_9 = 'Dict'
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Optional'
    var_13 = module_1.Load()
    var_14 = 'other_module'
    var_15 = module_1.Load()
    var_16 = module_1.Load()
    var_17 = 'module'
    var_18 = module_1.Load()
    var_19 = 'submodule'
    var_20 = module_1.Load()
    var_21 = 'Type'
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = 'Union'
    var_25 = module_1.Load()



# Parsed testcases at query #15
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = 'other'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = module_1.Load()
    var_12 = 'Dict'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = 'Optional'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'foo'
    var_19 = module_1.Load()
    var_20 = 'bar'
    var_21 = module_1.Load()
    var_22 = 'collections'
    var_23 = module_1.Load()
    var_24 = module_1.Load()



# Parsed testcases at query #16
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = 'other'
    var_9 = module_1.Load()
    var_10 = module_1.Load()
    var_11 = 'not_a_name'
    var_12 = 'attr'
    var_13 = module_1.Load()
    var_14 = module_1.Load()
    var_15 = 'Optional'
    var_16 = module_1.Load()
    var_17 = module_1.Load()
    var_18 = 'Union'
    var_19 = module_1.Load()
    var_20 = module_1.Load()
    var_21 = 'Dict'
    var_22 = module_1.Load()
    var_23 = module_1.Load()
    var_24 = 'sub'
    var_25 = module_1.Load()
    var_26 = module_1.Load()



# Parsed testcases at query #17
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'typing.List'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'not valid python @@@@'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = ''
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = 'test_module.MyType'
    var_26 = {var_25: var_13}
    var_27 = module_0.Resolver(var_1, var_26)
    var_28 = 'MyType'
    var_29 = module_1.Constant()
    var_30 = var_27.visit_Constant(var_29)
    var_31 = {}
    var_32 = 'T'
    var_33 = module_0.Resolver(var_1, var_31, var_32)
    var_34 = module_1.Constant()
    var_35 = var_33.visit_Constant(var_34)



# Parsed testcases at query #18
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass'
    var_4 = '\nclass TestClass(BaseClass, AnotherBase):\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.bases
    var_9 = var_7.body
    var_10 = var_1.class_api(var_2, var_3, var_8, var_9)
    var_11 = 'test_module.TestEnum'
    var_12 = '\nclass TestEnum(enum.Enum):\n    MEMBER1 = 1\n    MEMBER2 = 2\n'
    var_13 = module_1.parse(var_12)
    var_14 = var_13.body[var_6]
    var_15 = var_14.bases
    var_16 = var_14.body
    var_17 = var_1.class_api(var_2, var_11, var_15, var_16)
    var_18 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    another_public: float = 3.14\n'
    var_19 = module_1.parse(var_18)
    var_20 = var_19.body[var_6]
    var_21 = var_20.bases
    var_22 = var_20.body
    var_23 = var_1.class_api(var_2, var_3, var_21, var_22)
    var_24 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr1\n'
    var_25 = module_1.parse(var_24)
    var_26 = var_25.body[var_6]
    var_27 = var_26.bases
    var_28 = var_26.body
    var_29 = var_1.class_api(var_2, var_3, var_27, var_28)
    var_30 = 'class TestClass:\n    pass'
    var_31 = module_1.parse(var_30)
    var_32 = var_31.body[var_6]
    var_33 = var_32.bases
    var_34 = var_32.body
    var_35 = var_1.class_api(var_2, var_3, var_33, var_34)



# Parsed testcases at query #19
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
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
    var_19 = var_18.op
    var_20 = 'test_module.Optional'
    var_21 = 'typing.Optional'
    var_22 = {var_20: var_21}
    var_23 = module_0.Resolver(var_1, var_22)
    var_24 = 'Optional'
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = module_1.Load()
    var_30 = module_1.Subscript()
    var_31 = var_23.visit_Subscript(var_30)
    var_32 = var_31.op
    var_33 = var_31.right
    var_34 = 'test_module.List'
    var_35 = 'typing.List'
    var_36 = {var_34: var_35}
    var_37 = module_0.Resolver(var_1, var_36)
    var_38 = 'List'
    var_39 = module_1.Load()
    var_40 = module_1.Name()
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = module_1.Load()
    var_44 = module_1.Subscript()
    var_45 = var_37.visit_Subscript(var_44)
    var_46 = {}
    var_47 = module_0.Resolver(var_1, var_46)
    var_48 = 'typing'
    var_49 = module_1.Load()
    var_50 = module_1.Name()
    var_51 = module_1.Load()
    var_52 = module_1.Attribute()
    var_53 = module_1.Load()
    var_54 = module_1.Name()
    var_55 = module_1.Load()
    var_56 = module_1.Subscript()
    var_57 = var_47.visit_Subscript(var_56)
    var_58 = 'test_module.Union'
    var_59 = 'typing.Union'
    var_60 = {var_58: var_59}
    var_61 = module_0.Resolver(var_1, var_60)
    var_62 = module_1.Load()
    var_63 = module_1.Name()
    var_64 = module_1.Load()
    var_65 = module_1.Name()
    var_66 = module_1.Load()
    var_67 = module_1.Subscript()
    var_68 = var_61.visit_Subscript(var_67)
    var_69 = {var_58: var_59}
    var_70 = module_0.Resolver(var_1, var_69)
    var_71 = module_1.Load()
    var_72 = module_1.Name()
    var_73 = module_1.Load()
    var_74 = module_1.Name()
    var_75 = module_1.Load()
    var_76 = module_1.Name()
    var_77 = 'float'
    var_78 = module_1.Load()
    var_79 = module_1.Name()
    var_80 = [var_74, var_76, var_79]
    var_81 = module_1.Load()
    var_82 = module_1.Tuple()
    var_83 = module_1.Load()
    var_84 = module_1.Subscript()
    var_85 = var_70.visit_Subscript(var_84)
    var_86 = var_85.op
    var_87 = {}
    var_88 = module_0.Resolver(var_1, var_87)
    var_89 = 'CustomType'
    var_90 = module_1.Load()
    var_91 = module_1.Name()
    var_92 = module_1.Load()
    var_93 = module_1.Name()
    var_94 = module_1.Load()
    var_95 = module_1.Subscript()
    var_96 = var_88.visit_Subscript(var_95)



# Parsed testcases at query #20
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Name method.'
    var_1 = 'module'
    var_2 = {}
    var_3 = 'T'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = module_1.Load()
    var_6 = {}
    var_7 = module_0.Resolver(var_1, var_6)
    var_8 = 'SomeName'
    var_9 = module_1.Load()
    var_10 = 'module.OldName'
    var_11 = 'NewName'
    var_12 = {var_10: var_11}
    var_13 = module_0.Resolver(var_1, var_12)
    var_14 = 'OldName'
    var_15 = module_1.Load()
    var_16 = 'module.T'
    var_17 = 'module.TypeVar'
    var_18 = "typing.TypeVar('T')"
    var_19 = 'typing.TypeVar'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.Resolver(var_1, var_20)
    var_22 = module_1.Load()
    var_23 = 'module.SelfRef'
    var_24 = {var_23: var_23}
    var_25 = module_0.Resolver(var_1, var_24)
    var_26 = 'SelfRef'
    var_27 = module_1.Load()
    var_28 = 'module.ListInt'
    var_29 = 'List[int]'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_1, var_30)
    var_32 = 'ListInt'
    var_33 = module_1.Load()
    var_34 = 'pkg.module'
    var_35 = 'pkg.module.Type'
    var_36 = 'str'
    var_37 = {var_35: var_36}
    var_38 = module_0.Resolver(var_34, var_37)
    var_39 = 'Type'
    var_40 = module_1.Load()
    var_41 = ''
    var_42 = 'int'
    var_43 = {var_39: var_42}
    var_44 = module_0.Resolver(var_41, var_43)
    var_45 = module_1.Load()



# Parsed testcases at query #21
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Name method.'
    var_1 = 'module'
    var_2 = {}
    var_3 = 'T'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = module_1.Load()
    var_6 = {}
    var_7 = ''
    var_8 = module_0.Resolver(var_1, var_6, var_7)
    var_9 = 'SomeName'
    var_10 = module_1.Load()
    var_11 = 'module.MyType'
    var_12 = 'int'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13, var_7)
    var_15 = 'MyType'
    var_16 = module_1.Load()
    var_17 = {var_11: var_11}
    var_18 = module_0.Resolver(var_1, var_17, var_7)
    var_19 = module_1.Load()
    var_20 = 'module.T'
    var_21 = 'module.typing'
    var_22 = "typing.TypeVar('T')"
    var_23 = 'typing'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.Resolver(var_1, var_24, var_7)
    var_26 = module_1.Load()
    var_27 = 'pkg.module'
    var_28 = 'pkg.module.Type'
    var_29 = 'str'
    var_30 = {var_28: var_29}
    var_31 = module_0.Resolver(var_27, var_30, var_7)
    var_32 = 'Type'
    var_33 = module_1.Load()
    var_34 = {}
    var_35 = 'GenericType'
    var_36 = module_0.Resolver(var_1, var_34, var_35)
    var_37 = module_1.Load()
    var_38 = 'module.X'
    var_39 = {var_38: var_12}
    var_40 = 'X'
    var_41 = module_0.Resolver(var_1, var_39, var_40)
    var_42 = module_1.Load()



# Parsed testcases at query #22
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'Test walk_body function.'
    var_1 = 'x = 1\ny = 2'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body
    var_4 = module_1.walk_body(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = var_5[var_9]
    var_11 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_12 = module_0.parse(var_11)
    var_13 = var_12.body
    var_14 = module_1.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_18 = module_0.parse(var_17)
    var_19 = var_18.body
    var_20 = module_1.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3'
    var_24 = module_0.parse(var_23)
    var_25 = var_24.body
    var_26 = module_1.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept KeyError:\n    z = 3'
    var_30 = module_0.parse(var_29)
    var_31 = var_30.body
    var_32 = module_1.walk_body(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 3
    var_35 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3'
    var_36 = module_0.parse(var_35)
    var_37 = var_36.body
    var_38 = module_1.walk_body(var_37)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 3
    var_41 = []
    var_42 = module_1.walk_body(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 0
    var_45 = 'x = 1\nif True:\n    y = 2\nz = 3'
    var_46 = module_0.parse(var_45)
    var_47 = var_46.body
    var_48 = module_1.walk_body(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 3
    var_51 = var_49[var_7]
    var_52 = var_49[var_9]
    var_53 = 2
    var_54 = var_49[var_53]
    var_55 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\n    z = 3'
    var_56 = module_0.parse(var_55)
    var_57 = var_56.body
    var_58 = module_1.walk_body(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 3



# Parsed testcases at query #23
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test_module'
    var_2 = 'test_module.Union'
    var_3 = 'typing.Union'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 0
    var_7 = 'Union[int, str]'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = 'test_module.Optional'
    var_12 = 'typing.Optional'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'Optional[int]'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_6]
    var_18 = var_17.value
    var_19 = 'test_module.Dict'
    var_20 = 'typing.Dict'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'Dict[str, int]'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_6]
    var_26 = var_25.value
    var_27 = {}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'some_func()[int]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_6]
    var_32 = var_31.value
    var_33 = {}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = 'List[int]'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_6]
    var_38 = var_37.value
    var_39 = {var_2: var_3}
    var_40 = module_0.Resolver(var_1, var_39)
    var_41 = 'Union[int, str, float]'
    var_42 = module_1.parse(var_41)
    var_43 = var_42.body[var_6]
    var_44 = var_43.value



# Parsed testcases at query #24
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test'
    var_2 = 'test.Union'
    var_3 = 'typing.Union'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 'Union[int, str]'
    var_7 = 0
    var_8 = module_1.parse(var_6)
    var_9 = var_8.body[var_7]
    var_10 = var_9.value
    var_11 = 'test.Optional'
    var_12 = 'typing.Optional'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'Optional[int]'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_7]
    var_18 = var_17.value
    var_19 = 'test.Dict'
    var_20 = 'typing.Dict'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'Dict[str, int]'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_7]
    var_26 = var_25.value
    var_27 = {}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'some_module.Union[int, str]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_7]
    var_32 = var_31.value
    var_33 = {var_2: var_3}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = 'Union[int, str, float]'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_7]
    var_38 = var_37.value
    var_39 = {var_2: var_3}
    var_40 = module_0.Resolver(var_1, var_39)
    var_41 = {}
    var_42 = module_0.Resolver(var_1, var_41)
    var_43 = 'Union'
    var_44 = module_1.Load()
    var_45 = 'int'
    var_46 = module_1.Load()
    var_47 = module_1.Load()
    var_48 = {}
    var_49 = module_0.Resolver(var_1, var_48)
    var_50 = 'List[int]'
    var_51 = module_1.parse(var_50)
    var_52 = var_51.body[var_7]
    var_53 = var_52.value



# Parsed testcases at query #25
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass'
    var_4 = '\nclass TestClass:\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.bases
    var_9 = var_7.body
    var_10 = var_1.class_api(var_2, var_3, var_8, var_9)
    var_11 = '\nclass TestClass(BaseClass):\n    pass\n'
    var_12 = module_0.Parser()
    var_13 = module_1.parse(var_11)
    var_14 = var_13.body[var_6]
    var_15 = var_14.bases
    var_16 = var_14.body
    var_17 = var_12.class_api(var_2, var_3, var_15, var_16)
    var_18 = '\nclass TestClass:\n    attr1: int\n    attr2: str = "default"\n'
    var_19 = module_0.Parser()
    var_20 = module_1.parse(var_18)
    var_21 = var_20.body[var_6]
    var_22 = var_21.bases
    var_23 = var_21.body
    var_24 = var_19.class_api(var_2, var_3, var_22, var_23)
    var_25 = '\nclass TestEnum(enum.Enum):\n    VALUE1 = 1\n    VALUE2 = 2\n'
    var_26 = module_0.Parser()
    var_27 = module_1.parse(var_25)
    var_28 = var_27.body[var_6]
    var_29 = var_28.bases
    var_30 = var_28.body
    var_31 = var_26.class_api(var_2, var_3, var_29, var_30)
    var_32 = '\nclass TestClass:\n    attr1: int\n    del attr1\n'
    var_33 = module_0.Parser()
    var_34 = module_1.parse(var_32)
    var_35 = var_34.body[var_6]
    var_36 = var_35.bases
    var_37 = var_35.body
    var_38 = var_33.class_api(var_2, var_3, var_36, var_37)
    var_39 = '\nclass TestClass:\n    _private: int\n    public: str\n'
    var_40 = module_0.Parser()
    var_41 = module_1.parse(var_39)
    var_42 = var_41.body[var_6]
    var_43 = var_42.bases
    var_44 = var_42.body
    var_45 = var_40.class_api(var_2, var_3, var_43, var_44)



# Parsed testcases at query #26
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'Test const_type function with various AST nodes.'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'int'
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'float'
    var_7 = 'hello'
    var_8 = module_0.Constant()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'str'
    var_10 = True
    var_11 = module_0.Constant()
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'bool'
    var_13 = None
    var_14 = module_0.Constant()
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'NoneType'
    var_16 = module_0.Constant()
    var_17 = 2
    var_18 = module_0.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_0.Load()
    var_21 = module_0.Tuple()
    var_22 = module_1.const_type(var_21)
    assert var_22 == 'tuple[int, int]'
    var_23 = module_0.Constant()
    var_24 = 'str'
    var_25 = module_0.Constant()
    var_26 = [var_23, var_25]
    var_27 = module_0.Load()
    var_28 = module_0.Tuple()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'tuple[Any, Any]'
    var_30 = []
    var_31 = module_0.Load()
    var_32 = module_0.Tuple()
    var_33 = module_1.const_type(var_32)
    assert var_33 == 'tuple'
    var_34 = module_0.Constant()
    var_35 = module_0.Constant()
    var_36 = [var_34, var_35]
    var_37 = module_0.Load()
    var_38 = module_0.List()
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'list[int, int]'
    var_40 = module_0.Constant()
    var_41 = module_0.Constant()
    var_42 = [var_40, var_41]
    var_43 = module_0.Load()
    var_44 = module_0.List()
    var_45 = module_1.const_type(var_44)
    assert var_45 == 'list[Any, Any]'
    var_46 = []
    var_47 = module_0.Load()
    var_48 = module_0.List()
    var_49 = module_1.const_type(var_48)
    assert var_49 == 'list'
    var_50 = module_0.Constant()
    var_51 = module_0.Constant()
    var_52 = [var_50, var_51]
    var_53 = module_0.Set()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'set[int, int]'
    var_55 = module_0.Constant()
    var_56 = module_0.Constant()
    var_57 = [var_55, var_56]
    var_58 = module_0.Set()
    var_59 = module_1.const_type(var_58)
    assert var_59 == 'set[Any, Any]'
    var_60 = []
    var_61 = module_0.Set()
    var_62 = module_1.const_type(var_61)
    assert var_62 == 'set'
    var_63 = 'a'
    var_64 = module_0.Constant()
    var_65 = [var_64]
    var_66 = module_0.Constant()
    var_67 = [var_66]
    var_68 = module_0.Dict()
    var_69 = module_1.const_type(var_68)
    assert var_69 == 'dict[str, int]'
    var_70 = module_0.Constant()
    var_71 = module_0.Constant()
    var_72 = [var_70, var_71]
    var_73 = module_0.Constant()
    var_74 = module_0.Constant()
    var_75 = [var_73, var_74]
    var_76 = module_0.Dict()
    var_77 = module_1.const_type(var_76)
    assert var_77 == 'dict[Any, int]'
    var_78 = module_0.Constant()
    var_79 = 'b'
    var_80 = module_0.Constant()
    var_81 = [var_78, var_80]
    var_82 = module_0.Constant()
    var_83 = module_0.Constant()
    var_84 = [var_82, var_83]
    var_85 = module_0.Dict()
    var_86 = module_1.const_type(var_85)
    assert var_86 == 'dict[str, Any]'
    var_87 = []
    var_88 = []
    var_89 = module_0.Dict()
    var_90 = module_1.const_type(var_89)
    assert var_90 == 'dict'
    var_91 = 'int'
    var_92 = module_0.Load()
    var_93 = module_0.Name()
    var_94 = []
    var_95 = []
    var_96 = module_0.Call(*var_94)
    var_97 = module_1.const_type(var_96)
    assert var_97 == 'int'
    var_98 = module_0.Load()
    var_99 = module_0.Name()
    var_100 = []
    var_101 = []
    var_102 = module_0.Call(*var_100)
    var_103 = module_1.const_type(var_102)
    assert var_103 == 'str'
    var_104 = 'bool'
    var_105 = module_0.Load()
    var_106 = module_0.Name()
    var_107 = []
    var_108 = []
    var_109 = module_0.Call(*var_107)
    var_110 = module_1.const_type(var_109)
    assert var_110 == 'bool'
    var_111 = 'float'
    var_112 = module_0.Load()
    var_113 = module_0.Name()
    var_114 = []
    var_115 = []
    var_116 = module_0.Call(*var_114)
    var_117 = module_1.const_type(var_116)
    assert var_117 == 'float'
    var_118 = 'complex'
    var_119 = module_0.Load()
    var_120 = module_0.Name()
    var_121 = []
    var_122 = []
    var_123 = module_0.Call(*var_121)
    var_124 = module_1.const_type(var_123)
    assert var_124 == 'complex'
    var_125 = 'unknown_func'
    var_126 = module_0.Load()
    var_127 = module_0.Name()
    var_128 = []
    var_129 = []
    var_130 = module_0.Call(*var_128)
    var_131 = module_1.const_type(var_130)
    var_132 = module_0.Constant()
    var_133 = module_0.BitOr()
    var_134 = module_0.Constant()
    var_135 = module_0.BinOp()
    var_136 = module_1.const_type(var_135)
    var_137 = 'x'
    var_138 = module_0.Load()
    var_139 = module_0.Name()
    var_140 = module_1.const_type(var_139)
    var_141 = module_0.Constant()
    var_142 = [var_13, var_141]
    var_143 = module_0.Load()
    var_144 = module_0.Tuple()
    var_145 = module_1.const_type(var_144)
    assert var_145 == 'tuple'
    var_146 = module_0.Constant()
    var_147 = [var_146, var_13]
    var_148 = module_0.Load()
    var_149 = module_0.List()
    var_150 = module_1.const_type(var_149)
    assert var_150 == 'list'



# Parsed testcases at query #27
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.imports method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'os'
    var_4 = None
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 'test_module.submodule'
    var_8 = 'path'
    var_9 = 0
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'dd'
    var_13 = 'sibling'
    var_14 = 'func'
    var_15 = 1
    var_16 = 'package.subpackage.module'
    var_17 = 'other'
    var_18 = 'item'
    var_19 = 2
    var_20 = 'typing'
    var_21 = 'List'
    var_22 = 'Dict'
    var_23 = 'Optional'
    var_24 = 'Opt'



# Parsed testcases at query #28
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Name method.'
    var_1 = 'mymodule'
    var_2 = {}
    var_3 = 'MyType'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = module_1.Load()
    var_6 = {}
    var_7 = ''
    var_8 = module_0.Resolver(var_1, var_6, var_7)
    var_9 = 'SomeName'
    var_10 = module_1.Load()
    var_11 = 'mymodule.OldName'
    var_12 = 'NewName'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'OldName'
    var_16 = module_1.Load()
    var_17 = 'mymodule.TypeVar'
    var_18 = 'mymodule.T'
    var_19 = 'typing.TypeVar'
    var_20 = "TypeVar('T')"
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'T'
    var_24 = module_1.Load()
    var_25 = 'mymodule.Self'
    var_26 = 'Self'
    var_27 = {var_25: var_26}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = module_1.Load()
    var_30 = 'mymodule.MyList'
    var_31 = 'list[int]'
    var_32 = {var_30: var_31}
    var_33 = module_0.Resolver(var_1, var_32)
    var_34 = 'MyList'
    var_35 = module_1.Load()
    var_36 = 'mymodule.Other'
    var_37 = 'str'
    var_38 = {var_36: var_37}
    var_39 = module_0.Resolver(var_1, var_38)
    var_40 = 'UnknownName'
    var_41 = module_1.Load()
    var_42 = 'parent.child.Name'
    var_43 = 'int'
    var_44 = {var_42: var_43}
    var_45 = 'parent.child'
    var_46 = module_0.Resolver(var_45, var_44)
    var_47 = 'Name'
    var_48 = module_1.Load()



# Parsed testcases at query #29
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 'Optional'
    var_14 = module_1.Load()
    var_15 = module_1.Attribute()
    var_16 = var_3.visit_Attribute(var_15)
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = 'Dict'
    var_20 = module_1.Load()
    var_21 = module_1.Attribute()
    var_22 = var_3.visit_Attribute(var_21)
    var_23 = 'other_module'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = 'SomeClass'
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = var_3.visit_Attribute(var_28)
    var_30 = var_29.value
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = 'nested'
    var_34 = module_1.Load()
    var_35 = module_1.Attribute()
    var_36 = 'Inner'
    var_37 = module_1.Load()
    var_38 = module_1.Attribute()
    var_39 = var_3.visit_Attribute(var_38)
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = 'Union'
    var_43 = module_1.Load()
    var_44 = module_1.Attribute()
    var_45 = var_3.visit_Attribute(var_44)
    var_46 = module_1.Load()
    var_47 = module_1.Name()
    var_48 = 'Callable'
    var_49 = module_1.Load()
    var_50 = module_1.Attribute()
    var_51 = var_3.visit_Attribute(var_50)



# Parsed testcases at query #30
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Unit test for Parser.parse method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with nested module structure.'
    var_1 = module_0.Parser()
    var_2 = 'pkg.submodule'
    var_3 = '\ndef func(): pass\n'
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with type comments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '\nx = 42  # type: int\ny = "hello"  # type: str\n'
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with empty script.'
    var_1 = module_0.Parser()
    var_2 = 'empty_module'
    var_3 = ''
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with async function.'
    var_1 = module_0.Parser()
    var_2 = 'async_module'
    var_3 = "\nasync def async_func():\n    '''Async function.'''\n    pass\n"
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with decorated functions.'
    var_1 = module_0.Parser()
    var_2 = 'decorated_module'
    var_3 = "\nfrom functools import wraps\n\n@wraps\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with nested class definitions.'
    var_1 = module_0.Parser()
    var_2 = 'nested_module'
    var_3 = "\nclass OuterClass:\n    '''Outer class.'''\n    \n    class InnerClass:\n        '''Inner class.'''\n        pass\n"
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse with annotated assignments.'
    var_1 = module_0.Parser()
    var_2 = 'annotated_module'
    var_3 = '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n'
    var_4 = var_1.parse(var_2, var_3)



# Parsed testcases at query #31
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'Test const_type function with various AST node types.'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'int'
    var_4 = 3.14
    var_5 = module_0.Constant()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'float'
    var_7 = 'hello'
    var_8 = module_0.Constant()
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'str'
    var_10 = True
    var_11 = module_0.Constant()
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'bool'
    var_13 = None
    var_14 = module_0.Constant()
    var_15 = module_1.const_type(var_14)
    assert var_15 == 'NoneType'
    var_16 = module_0.Constant()
    var_17 = 2
    var_18 = module_0.Constant()
    var_19 = [var_16, var_18]
    var_20 = module_0.Load()
    var_21 = module_0.Tuple()
    var_22 = module_1.const_type(var_21)
    assert var_22 == 'tuple[int, int]'
    var_23 = module_0.Constant()
    var_24 = 'str'
    var_25 = module_0.Constant()
    var_26 = [var_23, var_25]
    var_27 = module_0.Load()
    var_28 = module_0.Tuple()
    var_29 = module_1.const_type(var_28)
    assert var_29 == 'tuple[Any, Any]'
    var_30 = module_0.Constant()
    var_31 = module_0.Constant()
    var_32 = [var_30, var_31]
    var_33 = module_0.Load()
    var_34 = module_0.List()
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'list[int]'
    var_36 = module_0.Constant()
    var_37 = module_0.Constant()
    var_38 = [var_36, var_37]
    var_39 = module_0.Load()
    var_40 = module_0.List()
    var_41 = module_1.const_type(var_40)
    assert var_41 == 'list[Any]'
    var_42 = module_0.Constant()
    var_43 = module_0.Constant()
    var_44 = [var_42, var_43]
    var_45 = module_0.Set()
    var_46 = module_1.const_type(var_45)
    assert var_46 == 'set[int]'
    var_47 = []
    var_48 = module_0.Load()
    var_49 = module_0.Tuple()
    var_50 = module_1.const_type(var_49)
    assert var_50 == 'tuple'
    var_51 = []
    var_52 = module_0.Load()
    var_53 = module_0.List()
    var_54 = module_1.const_type(var_53)
    assert var_54 == 'list'
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
    var_67 = 'val'
    var_68 = module_0.Constant()
    var_69 = [var_68]
    var_70 = module_0.Dict()
    var_71 = module_1.const_type(var_70)
    assert var_71 == 'dict[str, str]'
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
    var_83 = module_0.Load()
    var_84 = module_0.Name()
    var_85 = []
    var_86 = []
    var_87 = module_0.Call(*var_85)
    var_88 = module_1.const_type(var_87)
    assert var_88 == 'str'
    var_89 = 'bool'
    var_90 = module_0.Load()
    var_91 = module_0.Name()
    var_92 = []
    var_93 = []
    var_94 = module_0.Call(*var_92)
    var_95 = module_1.const_type(var_94)
    assert var_95 == 'bool'
    var_96 = 'float'
    var_97 = module_0.Load()
    var_98 = module_0.Name()
    var_99 = []
    var_100 = []
    var_101 = module_0.Call(*var_99)
    var_102 = module_1.const_type(var_101)
    assert var_102 == 'float'
    var_103 = 'complex'
    var_104 = module_0.Load()
    var_105 = module_0.Name()
    var_106 = []
    var_107 = []
    var_108 = module_0.Call(*var_106)
    var_109 = module_1.const_type(var_108)
    assert var_109 == 'complex'
    var_110 = 'custom_func'
    var_111 = module_0.Load()
    var_112 = module_0.Name()
    var_113 = []
    var_114 = []
    var_115 = module_0.Call(*var_113)
    var_116 = module_1.const_type(var_115)
    var_117 = module_0.Constant()
    var_118 = module_0.BitOr()
    var_119 = module_0.Constant()
    var_120 = module_0.BinOp()
    var_121 = module_1.const_type(var_120)
    var_122 = 'x'
    var_123 = module_0.Load()
    var_124 = module_0.Name()
    var_125 = module_1.const_type(var_124)
    var_126 = module_0.Constant()
    var_127 = [var_126, var_13]
    var_128 = module_0.Load()
    var_129 = module_0.Tuple()
    var_130 = module_1.const_type(var_129)
    assert var_130 == ''
    var_131 = module_0.Constant()
    var_132 = [var_131, var_13]
    var_133 = module_0.Load()
    var_134 = module_0.List()
    var_135 = module_1.const_type(var_134)
    assert var_135 == ''



# Parsed testcases at query #32
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'module'
    var_2 = 'module.Union'
    var_3 = 'typing.Union'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 0
    var_7 = 'Union[int, str]'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = 'module.Optional'
    var_12 = 'typing.Optional'
    var_13 = {var_11: var_12}
    var_14 = module_0.Resolver(var_1, var_13)
    var_15 = 'Optional[int]'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_6]
    var_18 = var_17.value
    var_19 = 'module.Dict'
    var_20 = 'typing.Dict'
    var_21 = {var_19: var_20}
    var_22 = module_0.Resolver(var_1, var_21)
    var_23 = 'Dict[str, int]'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_6]
    var_26 = var_25.value
    var_27 = {}
    var_28 = module_0.Resolver(var_1, var_27)
    var_29 = 'obj.Dict[int]'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_6]
    var_32 = var_31.value
    var_33 = {var_2: var_3}
    var_34 = module_0.Resolver(var_1, var_33)
    var_35 = 'Union[int]'
    var_36 = module_1.parse(var_35)
    var_37 = var_36.body[var_6]
    var_38 = var_37.value
    var_39 = {}
    var_40 = module_0.Resolver(var_1, var_39)
    var_41 = 'List[int]'
    var_42 = module_1.parse(var_41)
    var_43 = var_42.body[var_6]
    var_44 = var_43.value
    var_45 = {var_2: var_3}
    var_46 = module_0.Resolver(var_1, var_45)
    var_47 = 'Union[int, str, bool]'
    var_48 = module_1.parse(var_47)
    var_49 = var_48.body[var_6]
    var_50 = var_49.value
    var_51 = {var_11: var_12}
    var_52 = module_0.Resolver(var_1, var_51)
    var_53 = 'Optional[Optional[int]]'
    var_54 = module_1.parse(var_53)
    var_55 = var_54.body[var_6]
    var_56 = var_55.value



# Parsed testcases at query #33
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'List[str]'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = 'invalid syntax ]['
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'test_module.MyType'
    var_23 = {var_22: var_13}
    var_24 = module_0.Resolver(var_1, var_23)
    var_25 = 'MyType'
    var_26 = module_1.Constant()
    var_27 = var_24.visit_Constant(var_26)
    var_28 = 'Dict[str, int]'
    var_29 = module_1.Constant()
    var_30 = var_3.visit_Constant(var_29)
    var_31 = ''
    var_32 = module_1.Constant()
    var_33 = var_3.visit_Constant(var_32)
    var_34 = True
    var_35 = module_1.Constant()
    var_36 = var_3.visit_Constant(var_35)



# Parsed testcases at query #34
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass'
    var_4 = '\nclass MyClass(BaseClass, AnotherBase):\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.bases
    var_9 = var_7.body
    var_10 = var_1.class_api(var_2, var_3, var_8, var_9)
    var_11 = '\nclass MyEnum(enum.Enum):\n    OPTION_A = 1\n    OPTION_B = 2\n'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_6]
    var_14 = var_13.bases
    var_15 = var_13.body
    var_16 = var_1.class_api(var_2, var_3, var_14, var_15)
    var_17 = '\nclass MyClass:\n    public_attr: int = 5\n    _private_attr: str = "test"\n'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_6]
    var_20 = var_19.bases
    var_21 = var_19.body
    var_22 = var_1.class_api(var_2, var_3, var_20, var_21)
    var_23 = '\nclass MyClass:\n    attr1: int\n    attr2: str\n    del attr1\n'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_6]
    var_26 = var_25.bases
    var_27 = var_25.body
    var_28 = var_1.class_api(var_2, var_3, var_26, var_27)
    var_29 = '\nclass EmptyClass:\n    pass\n'
    var_30 = module_1.parse(var_29)
    var_31 = var_30.body[var_6]
    var_32 = var_31.bases
    var_33 = var_31.body
    var_34 = var_1.class_api(var_2, var_3, var_32, var_33)



# Parsed testcases at query #35
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.class_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass'
    var_4 = 'class Base: pass'
    var_5 = 0
    var_6 = module_1.parse(var_4)
    var_7 = var_6.body[var_5]
    var_8 = var_7.bases
    var_9 = []
    var_10 = var_1.class_api(var_2, var_3, var_8, var_9)
    var_11 = module_0.Parser()
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'
    var_14 = []
    var_15 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_5]
    var_18 = var_17.body
    var_19 = var_11.class_api(var_12, var_13, var_14, var_18)
    var_20 = module_0.Parser()
    var_21 = 'test_module'
    var_22 = 'test_module.TestEnum'
    var_23 = 'class TestEnum(enum.Enum): pass'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_5]
    var_26 = var_25.bases
    var_27 = '\nclass TestEnum(enum.Enum):\n    MEMBER1 = 1\n    MEMBER2 = 2\n    '
    var_28 = module_1.parse(var_27)
    var_29 = var_28.body[var_5]
    var_30 = var_29.body
    var_31 = 'test_module.enum'
    var_32 = 'enum'
    var_33 = var_20.class_api(var_21, var_22, var_26, var_30)
    var_34 = module_0.Parser()
    var_35 = 'test_module'
    var_36 = 'test_module.TestClass'
    var_37 = []
    var_38 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr1\n    '
    var_39 = module_1.parse(var_38)
    var_40 = var_39.body[var_5]
    var_41 = var_40.body
    var_42 = var_34.class_api(var_35, var_36, var_37, var_41)
    var_43 = module_0.Parser()
    var_44 = 'test_module'
    var_45 = 'test_module.TestClass'
    var_46 = []
    var_47 = '\nclass TestClass:\n    attr1 = 42  # type: int\n    attr2 = "hello"  # type: str\n    '
    var_48 = module_1.parse(var_47)
    var_49 = var_48.body[var_5]
    var_50 = var_49.body
    var_51 = var_43.class_api(var_44, var_45, var_46, var_50)
    var_52 = module_0.Parser()
    var_53 = 'test_module'
    var_54 = 'test_module.EmptyClass'
    var_55 = []
    var_56 = []
    var_57 = var_52.class_api(var_53, var_54, var_55, var_56)
    var_58 = var_52.doc[var_54]
    var_59 = '\n'



# Parsed testcases at query #36
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_ann method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = 0
    var_5 = 'int'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = var_7.value
    var_9 = module_1.arg()
    var_10 = 'y'
    var_11 = 'str'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_4]
    var_14 = var_13.value
    var_15 = module_1.arg()
    var_16 = 'return'
    var_17 = 'bool'
    var_18 = module_1.parse(var_17)
    var_19 = var_18.body[var_4]
    var_20 = var_19.value
    var_21 = module_1.arg()
    var_22 = [var_9, var_15, var_21]
    var_23 = False
    var_24 = False
    var_25 = var_1.func_ann(var_2, var_22, has_self=var_23, cls_method=var_24)
    var_26 = list(var_25)
    var_27 = 'self'
    var_28 = None
    var_29 = module_1.arg()
    var_30 = module_1.parse(var_5)
    var_31 = var_30.body[var_24]
    var_32 = var_31.value
    var_33 = module_1.arg()
    var_34 = 'None'
    var_35 = module_1.parse(var_34)
    var_36 = var_35.body[var_24]
    var_37 = var_36.value
    var_38 = module_1.arg()
    var_39 = [var_29, var_33, var_38]
    var_40 = True
    var_41 = False
    var_42 = var_1.func_ann(var_2, var_39, has_self=var_40, cls_method=var_41)
    var_43 = list(var_42)
    var_44 = 'cls'
    var_45 = 'type[TestClass]'
    var_46 = module_1.parse(var_45)
    var_47 = var_46.body[var_41]
    var_48 = var_47.value
    var_49 = module_1.arg()
    var_50 = module_1.parse(var_11)
    var_51 = var_50.body[var_41]
    var_52 = var_51.value
    var_53 = module_1.arg()
    var_54 = 'TestClass'
    var_55 = module_1.parse(var_54)
    var_56 = var_55.body[var_41]
    var_57 = var_56.value
    var_58 = module_1.arg()
    var_59 = [var_49, var_53, var_58]
    var_60 = var_1.func_ann(var_2, var_59, has_self=var_40, cls_method=var_40)
    var_61 = list(var_60)
    var_62 = module_1.arg()
    var_63 = module_1.arg()
    var_64 = module_1.arg()
    var_65 = [var_62, var_63, var_64]
    var_66 = False
    var_67 = False
    var_68 = var_1.func_ann(var_2, var_65, has_self=var_66, cls_method=var_67)
    var_69 = list(var_68)
    var_70 = module_1.parse(var_5)
    var_71 = var_70.body[var_67]
    var_72 = var_71.value
    var_73 = module_1.arg()
    var_74 = '*'
    var_75 = module_1.arg()
    var_76 = module_1.parse(var_11)
    var_77 = var_76.body[var_67]
    var_78 = var_77.value
    var_79 = module_1.arg()
    var_80 = module_1.arg()
    var_81 = [var_73, var_75, var_79, var_80]
    var_82 = False
    var_83 = False
    var_84 = var_1.func_ann(var_2, var_81, has_self=var_82, cls_method=var_83)
    var_85 = list(var_84)
    var_86 = 'MyClass'
    var_87 = module_1.parse(var_86)
    var_88 = var_87.body[var_83]
    var_89 = var_88.value
    var_90 = module_1.arg()
    var_91 = module_1.parse(var_5)
    var_92 = var_91.body[var_83]
    var_93 = var_92.value
    var_94 = module_1.arg()
    var_95 = module_1.arg()
    var_96 = [var_90, var_94, var_95]
    var_97 = False
    var_98 = var_1.func_ann(var_2, var_96, has_self=var_40, cls_method=var_97)
    var_99 = list(var_98)



# Parsed testcases at query #37
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Constant method.'
    var_1 = 'test_module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = 3.14
    var_8 = module_1.Constant()
    var_9 = var_3.visit_Constant(var_8)
    var_10 = None
    var_11 = module_1.Constant()
    var_12 = var_3.visit_Constant(var_11)
    var_13 = 'int'
    var_14 = module_1.Constant()
    var_15 = var_3.visit_Constant(var_14)
    var_16 = 'list[int]'
    var_17 = module_1.Constant()
    var_18 = var_3.visit_Constant(var_17)
    var_19 = '@#$%^'
    var_20 = module_1.Constant()
    var_21 = var_3.visit_Constant(var_20)
    var_22 = 'dict[str, int]'
    var_23 = module_1.Constant()
    var_24 = var_3.visit_Constant(var_23)
    var_25 = 'str'
    var_26 = module_1.Constant()
    var_27 = var_3.visit_Constant(var_26)
    var_28 = {}
    var_29 = 'T'
    var_30 = module_0.Resolver(var_1, var_28, var_29)
    var_31 = module_1.Constant()
    var_32 = var_30.visit_Constant(var_31)
    var_33 = 'test_module.MyType'
    var_34 = {var_33: var_13}
    var_35 = module_0.Resolver(var_1, var_34)
    var_36 = 'MyType'
    var_37 = module_1.Constant()
    var_38 = var_35.visit_Constant(var_37)



# Parsed testcases at query #38
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test Parser.compile method.'
    var_1 = module_0.Parser()
    var_2 = var_1.compile()
    var_3 = '\n'
    var_4 = True
    var_5 = module_0.Parser(toc=var_4)
    var_6 = var_5.compile()
    var_7 = False
    var_8 = module_0.Parser(var_4, var_4, var_7)
    var_9 = var_8.compile()
    var_10 = module_0.Parser(var_4, var_4, var_4)
    var_11 = var_10.compile()
    var_12 = module_0.Parser(var_7, var_4, var_7)
    var_13 = var_12.compile()
    var_14 = module_0.Parser(toc=var_7)
    var_15 = var_14.compile()
    var_16 = module_0.Parser(toc=var_4)
    var_17 = var_16.compile()



# Parsed testcases at query #39
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Resolver.visit_Attribute method.'
    var_1 = 'mymodule'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'typing'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Attribute()
    var_10 = var_3.visit_Attribute(var_9)
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 'Dict'
    var_14 = module_1.Load()
    var_15 = module_1.Attribute()
    var_16 = var_3.visit_Attribute(var_15)
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = 'MyClass'
    var_20 = module_1.Load()
    var_21 = module_1.Attribute()
    var_22 = var_3.visit_Attribute(var_21)
    var_23 = var_22.value
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = 'io'
    var_27 = module_1.Load()
    var_28 = module_1.Attribute()
    var_29 = 'StringIO'
    var_30 = module_1.Load()
    var_31 = module_1.Attribute()
    var_32 = var_3.visit_Attribute(var_31)
    var_33 = module_1.Load()
    var_34 = module_1.Name()
    var_35 = 'Optional'
    var_36 = module_1.Load()
    var_37 = module_1.Attribute()
    var_38 = var_3.visit_Attribute(var_37)
    var_39 = 'collections'
    var_40 = module_1.Load()
    var_41 = module_1.Name()
    var_42 = 'abc'
    var_43 = module_1.Load()
    var_44 = module_1.Attribute()
    var_45 = var_3.visit_Attribute(var_44)



# Parsed testcases at query #40
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.test_func'
    var_4 = '\ndef test_func(x: int, y: str = "default", *args, z: float = 1.0, **kwargs) -> bool:\n    pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.args
    var_10 = var_8.returns
    var_11 = False
    var_12 = False
    var_13 = var_1.func_api(var_2, var_3, var_9, var_10, has_self=var_11, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method with self parameter.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass.test_method'
    var_4 = '\nclass TestClass:\n    def test_method(self, x: int) -> str:\n        pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.body[var_6]
    var_10 = var_9.args
    var_11 = var_9.returns
    var_12 = True
    var_13 = False
    var_14 = var_1.func_api(var_2, var_3, var_10, var_11, has_self=var_12, cls_method=var_13)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method with classmethod.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.TestClass.test_classmethod'
    var_4 = '\nclass TestClass:\n    @classmethod\n    def test_classmethod(cls, x: int) -> str:\n        pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.body[var_6]
    var_10 = var_9.args
    var_11 = var_9.returns
    var_12 = True
    var_13 = var_1.func_api(var_2, var_3, var_10, var_11, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method with keyword-only arguments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.func_kwonly'
    var_4 = '\ndef func_kwonly(x: int, *, y: str, z: float = 1.0) -> None:\n    pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.args
    var_10 = var_8.returns
    var_11 = False
    var_12 = False
    var_13 = var_1.func_api(var_2, var_3, var_9, var_10, has_self=var_11, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method with positional-only arguments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.func_posonly'
    var_4 = '\ndef func_posonly(x: int, /, y: str) -> None:\n    pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.args
    var_10 = var_8.returns
    var_11 = False
    var_12 = False
    var_13 = var_1.func_api(var_2, var_3, var_9, var_10, has_self=var_11, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test Parser.func_api method with unannotated arguments.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'test_module.func_no_ann'
    var_4 = '\ndef func_no_ann(x, y):\n    pass\n'
    var_5 = var_1.parse(var_2, var_4)
    var_6 = 0
    var_7 = module_1.parse(var_4)
    var_8 = var_7.body[var_6]
    var_9 = var_8.args
    var_10 = var_8.returns
    var_11 = False
    var_12 = False
    var_13 = var_1.func_api(var_2, var_3, var_9, var_10, has_self=var_11, cls_method=var_12)



