####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.MyClass'
    var_5 = '## class MyClass\n\n*Full name:* `pkg.MyClass`\n<a id="pkg-MyClass"></a>\n\n'
    var_6 = 'MyClass'
    var_7 = 'BaseClass'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = []
    var_12 = 'ATTR_A'
    var_13 = 1
    var_14 = module_1.Constant()
    var_15 = 'int'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = 'pkg'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = [var_20]
    var_22 = 'MyEnum'
    var_23 = 'enum.Enum'
    var_24 = module_1.Load()
    var_25 = module_1.Name()
    var_26 = [var_25]
    var_27 = []
    var_28 = 'VAL_1'
    var_29 = module_1.Constant()
    var_30 = module_1.Load()
    var_31 = module_1.Name()
    var_32 = 'pkg'
    var_33 = 'MyEnum'
    var_34 = 'enum.Enum'
    var_35 = module_1.Load()
    var_36 = module_1.Name()
    var_37 = [var_36]
    var_38 = var_2.class_api(var_32, var_33, var_37, var_12)
    var_39 = []
    var_40 = []
    var_41 = module_1.Load()
    var_42 = module_1.Name()
    var_43 = [var_42]
    var_44 = module_1.Delete()
    var_45 = [var_44]
    var_46 = module_1.ClassDef()
    var_47 = []
    var_48 = var_46.body
    var_49 = var_2.class_api(var_18, var_32, var_47, var_48)
    var_50 = 'BaseNode'
    var_51 = 'Parent'
    var_52 = module_1.Load()
    var_53 = module_1.Name()
    var_54 = [var_53]
    var_55 = []
    var_56 = []
    var_57 = module_1.ClassDef()
    var_58 = module_1.Load()
    var_59 = module_1.Name()
    var_60 = [var_59]
    var_61 = var_57.body
    var_62 = var_2.class_api(var_34, var_50, var_60, var_61)



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'my_mod'
    var_1 = 'my_mod.target'
    var_2 = 'my_mod.actual_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 123
    var_6 = module_1.Constant()
    var_7 = var_4.visit_Constant(var_6)
    var_8 = '[unclosed bracket'
    var_9 = module_1.Constant()
    var_10 = var_4.visit_Constant(var_9)
    var_11 = 'target'
    var_12 = module_1.Constant()
    var_13 = var_4.visit_Constant(var_12)
    var_14 = '1 + 1'
    var_15 = module_1.Constant()
    var_16 = var_4.visit_Constant(var_15)
    var_17 = var_16.left
    var_18 = module_1.Constant()
    var_19 = var_4.visit_Constant(var_18)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''
    var_2 = 'This is a simple description.'
    var_3 = module_0.doctest(var_2)
    var_4 = '>>> 1 + 1\n2\n>>> 2 + 2'
    var_5 = '```python\n>>> 1 + 1\n2\n>>> 2 + 2\n```'
    var_6 = module_0.doctest(var_4)
    var_7 = 'Intro\n>>> 1\n1\nOutro'
    var_8 = 'Intro\n```python\n>>> 1\n1\n```\nOutro'
    var_9 = module_0.doctest(var_7)
    var_10 = 'Start\n>>> 1\n1\nMiddle\n>>> 2\n2\nEnd'
    var_11 = 'Start\n```python\n>>> 1\n1\n```\nMiddle\n```python\n>>> 2\n2\n```\nEnd'
    var_12 = module_0.doctest(var_10)
    var_13 = '>>> 1\n1\nResult is\npositive'
    var_14 = '```python\n>>> 1\n1\n```\nResult is\npositive'
    var_15 = module_0.doctest(var_13)
    var_16 = ">>> print('hi')"
    var_17 = "```python\n>>> print('hi')\n```"
    var_18 = module_0.doctest(var_16)



# Parsed testcases at query #4
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = '\nif True:\n    a = 1\nelse:\n    b = 2\n'
    var_11 = module_0.parse(var_10)
    var_12 = var_11.body
    var_13 = module_1.walk_body(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_14[var_6]
    var_17 = var_14[var_8]
    var_18 = 2
    var_19 = var_14[var_18]
    var_20 = '\ntry:\n    c = 3\nexcept ValueError:\n    d = 4\nelse:\n    e = 5\nfinally:\n    f = 6\n'
    var_21 = module_0.parse(var_20)
    var_22 = var_21.body
    var_23 = module_1.walk_body(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 5
    var_26 = var_24[var_6]
    var_27 = var_24[var_8]
    var_28 = var_24[var_18]
    var_29 = 3
    var_30 = var_24[var_29]
    var_31 = 4
    var_32 = var_24[var_31]
    var_33 = '\ntry:\n    if True:\n        z = 10\nexcept:\n    if False:\n        w = 20\n    else:\n        w = 30\n'
    var_34 = module_0.parse(var_33)
    var_35 = var_34.body
    var_36 = module_1.walk_body(var_35)
    var_37 = list(var_36)
    var_38 = [type(n) for n in var_37]
    var_39 = []
    var_40 = module_1.walk_body(var_39)
    var_41 = list(var_40)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    Since the method relies heavily on internal state (doc, root, level, etc.) \n    and private helper methods, we mock the complex logic to verify \n    the assembly of the final string.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg.module'
    var_4 = 'pkg.module.func'
    var_5 = 0
    var_6 = '# Module `pkg.module`\n<a id="pkg.module"></a>\n\n'
    var_7 = '## func()\n\n*Full name:* `pkg.module.func`\n<a id="pkg.module.func"></a>\n\n'
    var_8 = 'Module doc content.'
    var_9 = 'Function doc content.'
    var_10 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Tests compile without Table of Contents.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg.module'
    var_5 = '# Module `pkg.module`\n\n'
    var_6 = 'Doc'
    var_7 = var_3.compile()



# Parsed testcases at query #7
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'pep565'
    var_1 = 'typing.List'
    var_2 = 'typing.Dict'
    var_3 = 'list'
    var_4 = 'dict'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 0
    var_7 = 'typing.List[int]'
    var_8 = module_0.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = 'test'
    var_12 = {}
    var_13 = module_1.Resolver(var_11, var_12)
    var_14 = var_13.visit_Subscript(var_10)
    var_15 = var_14.value
    var_16 = 'typing.Union[int, str]'
    var_17 = module_0.parse(var_16)
    var_18 = var_17.body[var_6]
    var_19 = var_18.value
    var_20 = 'test.Union'
    var_21 = 'typing.Union'
    var_22 = {var_20: var_21}
    var_23 = module_1.Resolver(var_11, var_22)
    var_24 = var_23.visit_Subscript(var_19)
    var_25 = var_24.op
    var_26 = var_24.left
    var_27 = var_24.right
    var_28 = 'typing.Optional[int]'
    var_29 = module_0.parse(var_28)
    var_30 = var_29.body[var_6]
    var_31 = var_30.value
    var_32 = 'test.Optional'
    var_33 = 'typing.Optional'
    var_34 = {var_32: var_33}
    var_35 = module_1.Resolver(var_11, var_34)
    var_36 = var_35.visit_Subscript(var_31)
    var_37 = var_36.op
    var_38 = var_36.right
    var_39 = module_0.parse(var_7)
    var_40 = var_39.body[var_6]
    var_41 = var_40.value
    var_42 = 'test.List'
    var_43 = {var_42: var_1}
    var_44 = module_1.Resolver(var_11, var_43)
    var_45 = var_44.visit_Subscript(var_41)
    var_46 = var_45.value
    var_47 = 'List[int]'
    var_48 = module_0.parse(var_47)
    var_49 = var_48.body[var_6]
    var_50 = var_49.value
    var_51 = {}
    var_52 = module_1.Resolver(var_11, var_51)
    var_53 = var_52.visit_Subscript(var_50)
    var_54 = var_53.value



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Test the imports method of the Parser class.\n    Verifies that:\n    1. Standard imports (Import) are correctly mapped to the root.\n    2. From imports (ImportFrom) are correctly mapped using the module path.\n    3. Aliased imports (asname) are correctly handled.\n    4. Parent module logic is respected via the _m helper.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'my_package.sub_module'
    var_5 = 0
    var_6 = 'import os as system'
    var_7 = module_1.parse(var_6)
    var_8 = var_7.body[var_5]
    var_9 = var_3.imports(var_4, var_8)
    var_10 = 'from math import sqrt'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_5]
    var_13 = var_3.imports(var_4, var_12)
    var_14 = 'from . import utils'
    var_15 = module_1.parse(var_14)
    var_16 = var_15.body[var_5]
    var_17 = var_3.imports(var_4, var_16)
    var_18 = 'utils'
    var_19 = any(var_9)
    var_20 = var_3.alias
    var_21 = len(var_20)



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the class_api method of the Parser class.\n    It verifies that the method correctly processes class bases, \n    identifies Enums, and populates the doc/table with members/enums.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'mypkg'
    var_5 = 'ResolvedType'
    var_6 = lambda root, node: var_5
    var_7 = 'my_member'
    var_8 = 123
    var_9 = 'annotated_member'
    var_10 = 'Type'
    var_11 = 'val'
    var_12 = 'mypkg'
    var_13 = 'mypkg.MyClass'
    var_14 = []
    var_15 = []
    var_16 = var_3.class_api(var_12, var_13, var_14, var_15)
    var_17 = 'enum.MyEnumBase'
    var_18 = 'id'
    var_19 = 'Base'
    var_20 = lambda root, node: var_17 if hasattr(node, var_18) and node.id == var_19 else var_19
    var_21 = 'VAL_ONE'
    var_22 = 'mypkg'
    var_23 = 'mypkg.MyEnum'
    var_24 = []
    var_25 = var_3.class_api(var_22, var_23, var_14, var_24)
    var_26 = 'mypkg.MyClass'
    var_27 = 'OldDoc'
    var_28 = 'myppend'
    var_29 = 'mypkg.MyClass'
    var_30 = []
    var_31 = []
    var_32 = var_3.class_api(var_28, var_29, var_30, var_31)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Test the imports method of the Parser class.\n    Verifies that Import and ImportFrom nodes correctly populate the alias dictionary.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'os'
    var_5 = [var_1]
    var_6 = module_1.Import()
    var_7 = 'pkg.module'
    var_8 = 'sys'
    var_9 = 's'
    var_10 = 'pkg.sub'
    var_11 = 'func'
    var_12 = 'f'
    var_13 = [var_7]
    var_14 = 0
    var_15 = module_1.ImportFrom()
    var_16 = 'pkg.module'
    var_17 = var_3.imports(var_16, var_15)



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Test the globals method of the Parser class.\n    It should handle:\n    1. AnnAssign (Annotated Assignment) with resolution.\n    2. Assign (Simple Assignment) with type comments.\n    3. __all__ processing for imports.\n    4. Filtering non-target assignments.\n    '
    var_1 = 'my_module'
    var_2 = module_0.Parser()
    var_3 = 'MY_VAR'
    var_4 = None
    var_5 = module_1.Name()
    var_6 = 10
    var_7 = module_1.Constant()
    var_8 = 'int'
    var_9 = module_1.Name()
    var_10 = 1
    var_11 = module_1.AnnAssign()
    var_12 = var_2.globals(var_1, var_11)
    var_13 = module_0.Parser()
    var_14 = 'CONST_VAL'
    var_15 = None
    var_16 = module_1.Name()
    var_17 = [var_16]
    var_18 = 'hello'
    var_19 = module_1.Constant()
    var_20 = 'str'
    var_21 = module_1.Assign()
    var_22 = var_13.globals(var_21)
    var_23 = False
    var_24 = True
    var_25 = module_0.Parser()
    var_26 = set()
    var_27 = '__all__'
    var_28 = module_1.Name()
    var_29 = [var_28]
    var_30 = 'a.b'
    var_31 = module_1.Constant()
    var_32 = 'c.d'
    var_33 = module_1.Constant()
    var_34 = [var_31, var_33]
    var_35 = module_1.Tuple()
    var_36 = module_1.Assign()
    var_37 = var_25.globals(var_1, var_36)
    var_38 = module_0.Parser()
    var_39 = 1
    var_40 = module_1.Constant()
    var_41 = var_38.globals(var_1, var_40)
    var_42 = var_38.alias
    var_43 = len(var_42)
    assert var_43 == 0



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = 'sys.path'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'my_module.sub_module.func'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = 'module.__init__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = 'module.Class'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is True
    var_10 = '_private'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is False
    var_12 = 'module._private'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is False
    var_14 = 'module.sub._private'
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is False
    var_16 = 'module.__name__'
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = 'module.__str__'
    var_19 = module_0.is_public_family(var_18)
    assert var_19 is True
    var_20 = '__main__'
    var_21 = module_0.is_public_family(var_20)
    assert var_21 is True
    var_22 = ''
    var_23 = module_0.is_public_family(var_22)
    assert var_23 is True
    var_24 = '...'
    var_25 = module_0.is_public_family(var_24)
    assert var_25 is True



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Tests the `globals` method of the Parser class, covering:\n    1. AnnAssign (Annotated Assignment) with resolution.\n    2. Assign (Simple Assignment) with type comments.\n    3. Simple Assignment with constant type inference.\n    4. Handling of __all__ for importing submodules.\n    5. Ignoring non-target or non-assign nodes.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = set()
    var_6 = 'x'
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 10
    var_11 = module_1.Constant()
    var_12 = None
    var_13 = module_1.Load()
    var_14 = 'pkg'
    var_15 = 'y'
    var_16 = 20
    var_17 = module_1.Constant()
    var_18 = 'str'
    var_19 = 'pkg'
    var_20 = 'Z'
    var_21 = 30
    var_22 = module_1.Constant()
    var_23 = 'pkg'
    var_24 = '__all__'
    var_25 = 'sub'
    var_26 = module_1.Constant()
    var_27 = [var_26]
    var_28 = module_1.Load()
    var_29 = module_1.Tuple()
    var_30 = 'pkg'
    var_31 = module_1.Constant()
    var_32 = module_1.Expr()
    var_33 = var_3.alias
    var_34 = len(var_33)
    var_35 = var_3.globals(var_4, var_32)
    var_36 = var_3.alias
    var_37 = len(var_36)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    Since compile() relies heavily on the internal state of the Parser \n    (doc, docstring, root, level, etc.) being populated by parse(),\n    this test verifies that the compilation logic correctly assembles \n    the final string, handles the Table of Contents (TOC), and \n    filters/formats entries based on the internal state.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.mod'
    var_5 = 'pkg.mod.func'
    var_6 = 0
    var_7 = 2
    var_8 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_9 = '## mod\n<a id="pkg.mod"></a>\n\n'
    var_10 = '### func()\n\n'
    var_11 = 'Docstring for func.'
    var_12 = set()
    var_13 = set()
    var_14 = 'pkg.CONST'
    var_15 = 'int'
    var_16 = var_2.compile()
    var_17 = False
    var_18 = module_0.Parser(var_1, toc=var_17, level=var_1)
    var_19 = '# Module `pkg`\n\n'
    var_20 = set()
    var_21 = var_18.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Tests that non-public members are filtered out of the final output.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.private'
    var_6 = '# Module `pkg`\n\n'
    var_7 = '## private\n\n'
    var_8 = set()
    var_9 = var_3.compile()



# Parsed testcases at query #4
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Tests the `api` method of the `Parser` class by simulating the parsing\n    of a class and a function to verify documentation generation.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.mod'
    var_6 = set()
    var_7 = 'my_func'
    var_8 = []
    var_9 = 'x'
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = module_1.arg()
    var_14 = [var_13]
    var_15 = []
    var_16 = []
    var_17 = 10
    var_18 = module_1.Constant()
    var_19 = [var_18]
    var_20 = module_1.arguments(*var_14)
    var_21 = []
    var_22 = 'str'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = 'MyClass'
    var_26 = 'object'
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = [var_28]
    var_30 = []
    var_31 = 'ATTR'
    var_32 = module_1.Constant()
    var_33 = None
    var_34 = []
    var_35 = 'pkg'
    var_36 = ''
    var_37 = 'pkg.my_func'
    var_38 = 'pkg.MyClass'



# Parsed testcases at query #5
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'typing.List'
    var_1 = 'typing.Dict'
    var_2 = 'list'
    var_3 = 'dict'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 'typing.List[int]'
    var_7 = module_0.parse(var_6)
    var_8 = var_7.body[var_5]
    var_9 = var_8.value
    var_10 = 'mod'
    var_11 = {}
    var_12 = ''
    var_13 = module_1.Resolver(var_10, var_11, var_12)
    var_14 = var_13.visit_Subscript(var_9)
    var_15 = var_14.value
    var_16 = 'Union[int, str]'
    var_17 = module_0.parse(var_16)
    var_18 = var_17.body[var_5]
    var_19 = var_18.value
    var_20 = 'mod.Union'
    var_21 = 'typing.Union'
    var_22 = {var_20: var_21}
    var_23 = module_1.Resolver(var_10, var_22, var_12)
    var_24 = var_23.visit_Subscript(var_19)
    var_25 = var_24.op
    var_26 = var_24.left
    var_27 = var_24.right
    var_28 = 'Optional[int]'
    var_29 = module_0.parse(var_28)
    var_30 = var_29.body[var_5]
    var_31 = var_30.value
    var_32 = 'mod.Optional'
    var_33 = 'typing.Optional'
    var_34 = {var_32: var_33}
    var_35 = module_1.Resolver(var_10, var_34, var_12)
    var_36 = var_35.visit_Subscript(var_31)
    var_37 = var_36.op
    var_38 = var_36.right
    var_39 = 'List[int]'
    var_40 = module_0.parse(var_39)
    var_41 = var_40.body[var_5]
    var_42 = var_41.value
    var_43 = 'mod.List'
    var_44 = {var_43: var_0}
    var_45 = module_1.Resolver(var_10, var_44, var_12)
    var_46 = var_45.visit_Subscript(var_42)
    var_47 = var_46.value
    var_48 = 'MyType[int]'
    var_49 = module_0.parse(var_48)
    var_50 = var_49.body[var_5]
    var_51 = var_50.value
    var_52 = {}
    var_53 = module_1.Resolver(var_10, var_52, var_12)
    var_54 = var_53.visit_Subscript(var_51)
    var_55 = 'Other[int]'
    var_56 = module_0.parse(var_55)
    var_57 = var_56.body[var_5]
    var_58 = var_57.value
    var_59 = 'mod.Other'
    var_60 = 'typing.Other'
    var_61 = {var_59: var_60}
    var_62 = module_1.Resolver(var_10, var_61, var_12)
    var_63 = var_62.visit_Subscript(var_58)



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Test the compile method of the Parser class.\n    This test verifies that the compile method correctly:\n    1. Triggers alias substitution via __find_alias.\n    2. Generates a Table of Contents (TOC) if toc=True.\n    3. Iterates through sorted documentation entries.\n    4. Filters by public visibility.\n    5. Formats docstrings and handles constants.\n    6. Returns the final concatenated string.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.sub'
    var_5 = 0
    var_6 = '# Module `pkg`\n<a id="pkg"></a>\n\nContent pkg'
    var_7 = '## sub\n<a id="pkg.sub"></a>\n\nContent sub'
    var_8 = 'Docstring pkg'
    var_9 = 'Docstring sub'
    var_10 = set()
    var_11 = var_2.compile()



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = "\n    Tests the is_public method of the Parser class.\n    The method checks if a name is public based on:\n    1. If it's in self.imp, it must be the root or a child of a name in __all__.\n    2. If not in self.imp, it must follow is_public_family (not starting with underscore).\n    "
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg.module'
    var_5 = var_3.is_public(var_4)
    assert var_5 is True
    var_6 = 'pkg._private'
    var_7 = 'pkg.module.sub'
    var_8 = {var_7}
    var_9 = var_3.is_public(var_7)
    assert var_9 is True
    var_10 = 'pkg.module.other'
    var_11 = {var_10}
    var_12 = var_3.is_public(var_7)
    assert var_12 is False
    var_13 = {var_4}
    var_14 = var_3.is_public(var_4)
    assert var_14 is True
    var_15 = 'pkg.module.sub.attr'
    var_16 = {var_15}
    var_17 = 'content'
    var_18 = var_3.is_public(var_15)
    assert var_18 is True
    var_19 = {var_7}
    var_20 = var_3.is_public(var_7)
    assert var_20 is True



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    The test verifies that the method correctly aggregates documentation,\n    handles Table of Contents (TOC), processes aliases via __find_alias,\n    includes constants, and respects public visibility.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg.module'
    var_4 = 'pkg.module.Func'
    var_5 = 2
    var_6 = 'Func'
    var_7 = set()
    var_8 = 'pkg.module.CONST'
    var_9 = 'int'



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mymodule'
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
    var_11 = 'submodule'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'Class'
    var_15 = module_1.Load()
    var_16 = module_1.Attribute()
    var_17 = var_2.visit_Attribute(var_16)
    var_18 = 'a'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = 'b'
    var_22 = module_1.Load()
    var_23 = module_1.Attribute()
    var_24 = 'c'
    var_25 = module_1.Load()
    var_26 = module_1.Attribute()
    var_27 = var_2.visit_Attribute(var_26)
    var_28 = var_27.value



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'PEP585'
    var_1 = 'typing.List'
    var_2 = 'typing.Dict'
    var_3 = [var_1, var_2]
    var_4 = 'list'
    var_5 = 'dict'
    var_6 = {var_1: var_4, var_2: var_5}
    var_7 = 'my_mod.Union'
    var_8 = 'my_mod.Optional'
    var_9 = 'my_mod.List'
    var_10 = 'typing.Union'
    var_11 = 'typing.Optional'
    var_12 = {var_7: var_10, var_8: var_11, var_9: var_1}
    var_13 = 'my_mod'
    var_14 = module_0.Resolver(var_13, var_12)
    var_15 = 'typing'
    var_16 = module_1.Load()
    var_17 = module_1.Name()
    var_18 = 'List'
    var_19 = module_1.Load()
    var_20 = module_1.Attribute()
    var_21 = var_14.visit_Subscript(var_20)
    var_22 = 'Other'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = 'int'
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = module_1.Load()
    var_29 = module_1.Subscript()
    var_30 = var_14.visit_Subscript(var_29)
    var_31 = 'Union'
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = 'str'
    var_37 = module_1.Load()
    var_38 = module_1.Name()
    var_39 = [var_35, var_38]
    var_40 = module_1.Load()
    var_41 = module_1.Tuple()
    var_42 = module_1.Load()
    var_43 = module_1.Subscript()
    var_44 = 0
    var_45 = 'Union[int, str]'
    var_46 = module_1.parse(var_45)
    var_47 = var_46.body[var_44]
    var_48 = var_47.value
    var_49 = 'Optional[int]'
    var_50 = module_1.parse(var_49)
    var_51 = var_50.body[var_44]
    var_52 = var_51.value
    var_53 = module_1.Load()
    var_54 = module_1.Name()
    var_55 = module_1.Load()
    var_56 = module_1.Name()
    var_57 = [var_54, var_56]
    var_58 = module_1.Load()
    var_59 = module_1.Tuple()
    var_60 = 'Optional'
    var_61 = module_1.Load()
    var_62 = module_1.Name()
    var_63 = module_1.Load()
    var_64 = module_1.Name()



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    The test covers:\n    1. Alias substitution via __find_alias.\n    2. Table of contents generation.\n    3. Public vs Private name filtering.\n    4. Formatting of docstrings and constants.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.mod'
    var_5 = 'pkg.mod.func'
    var_6 = 0
    var_7 = 2
    var_8 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_9 = '# Module `pkg.mod`\n<a id="pkg.mod"></a>\n\n'
    var_10 = '## func()\n\n*Full name:* `pkg.mod.func`\n<a id="pkg.mod.func"></a>\n\n'
    var_11 = 'Docstring for func.'
    var_12 = set()



