####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Tests the Parser.api method to ensure it correctly populates \n    doc and docstring dictionaries for functions and classes.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_6 = 'my_func'
    var_7 = []
    var_8 = 'x'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = module_1.arg()
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = 10
    var_17 = module_1.Constant()
    var_18 = [var_17]
    var_19 = module_1.arguments(*var_13)
    var_20 = []
    var_21 = 'str'
    var_22 = module_1.Load()
    var_23 = module_1.Name()
    var_24 = 'MyClass'
    var_25 = 'Base'
    var_26 = module_1.Load()
    var_27 = module_1.Name()
    var_28 = [var_27]
    var_29 = []
    var_30 = 'ATTR'
    var_31 = module_1.Load()
    var_32 = module_1.Name()
    var_33 = 5
    var_34 = module_1.Constant()
    var_35 = None
    var_36 = []
    var_37 = 'pkg'
    var_38 = 'pkg.my_func'
    var_39 = 'my_func'
    var_40 = 'pkg.my_func.MyClass'



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test the is_public method of the Parser class.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.mod'
    var_6 = 'pkg.module'
    var_7 = var_3.is_public(var_6)
    assert var_7 is True
    var_8 = 'pkg._private'
    var_9 = var_3.is_public(var_8)
    assert var_9 is False
    var_10 = 'pkg.api'
    var_11 = var_3.is_public(var_10)
    assert var_11 is True
    var_12 = 'pkg.other'
    var_13 = var_3.is_public(var_10)
    assert var_13 is False
    var_14 = var_3.is_public(var_4)
    assert var_14 is True
    var_15 = 'pkg.sub'
    var_16 = var_3.is_public(var_15)
    assert var_16 is True
    var_17 = 'pkg.sub.child'
    var_18 = var_3.is_public(var_17)
    assert var_18 is True
    var_19 = 'pkg.hidden'
    var_20 = var_3.is_public(var_19)
    assert var_20 is False



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 0
    var_4 = 'typing.List'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = var_2.visit_Attribute(var_7)
    var_9 = var_8.ctx
    var_10 = {}
    var_11 = module_0.Resolver(var_0, var_10)
    var_12 = 'collections.abc.Iterable'
    var_13 = module_1.parse(var_12)
    var_14 = var_13.body[var_3]
    var_15 = var_14.value
    var_16 = var_11.visit_Attribute(var_15)
    var_17 = {}
    var_18 = module_0.Resolver(var_0, var_17)
    var_19 = 'a.b.c'
    var_20 = module_1.parse(var_19)
    var_21 = var_20.body[var_3]
    var_22 = var_21.value
    var_23 = var_18.visit_Attribute(var_22)
    var_24 = var_23.value



# Parsed testcases at query #4
#--------------------------


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1; y = 2'
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
    assert var_25 == 4
    var_26 = var_24[var_6]
    var_27 = var_24[var_8]
    var_28 = var_24[var_18]
    var_29 = 3
    var_30 = var_24[var_29]
    var_31 = '\ntry:\n    if True:\n        g = 7\nexcept:\n    pass\n'
    var_32 = module_0.parse(var_31)
    var_33 = var_32.body
    var_34 = module_1.walk_body(var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_35[var_6]



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Test the func_api method of the Parser class.\n    It tests various argument configurations: positional-only, args, varargs, \n    kwonlyargs, kwargs, and return annotations.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'



# Parsed testcases at query #6
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
    var_6 = [var_4, var_5]
    var_7 = 'my_mod'
    var_8 = {}
    var_9 = module_0.Resolver(var_7, var_8)
    var_10 = 0
    var_11 = 'typing.List[int]'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_10]
    var_14 = var_13.value
    var_15 = var_14
    var_16 = var_9.visit_Subscript(var_15)
    var_17 = var_16.value
    var_18 = 'my_mod.Union'
    var_19 = 'typing.Union'
    var_20 = {var_18: var_19}
    var_21 = module_0.Resolver(var_7, var_20)
    var_22 = 'Union[int, str]'
    var_23 = module_1.parse(var_22)
    var_24 = var_23.body[var_10]
    var_25 = var_24.value
    var_26 = var_21.visit_Subscript(var_25)
    var_27 = var_26.op
    var_28 = var_26.left
    var_29 = var_26.right
    var_30 = 'my_mod.Optional'
    var_31 = 'typing.Optional'
    var_32 = {var_30: var_31}
    var_33 = module_0.Resolver(var_7, var_32)
    var_34 = 'Optional[int]'
    var_35 = module_1.parse(var_34)
    var_36 = var_35.body[var_10]
    var_37 = var_36.value
    var_38 = var_33.visit_Subscript(var_37)
    var_39 = var_38.op
    var_40 = var_38.right
    var_41 = 'my_mod.List'
    var_42 = {var_41: var_1}
    var_43 = module_0.Resolver(var_7, var_42)
    var_44 = 'List[int]'
    var_45 = module_1.parse(var_44)
    var_46 = var_45.body[var_10]
    var_47 = var_46.value
    var_48 = var_43.visit_Subscript(var_47)
    var_49 = var_48.value
    var_50 = {}
    var_51 = module_0.Resolver(var_7, var_50)
    var_52 = module_1.parse(var_44)
    var_53 = var_52.body[var_10]
    var_54 = var_53.value
    var_55 = var_51.visit_Subscript(var_54)
    var_56 = 'my_mod.SomeType'
    var_57 = {var_56: var_19}
    var_58 = module_0.Resolver(var_7, var_57)
    var_59 = 'pkg.SomeType[int]'
    var_60 = module_1.parse(var_59)
    var_61 = var_60.body[var_10]
    var_62 = var_61.value
    var_63 = var_58.visit_Subscript(var_62)
    var_64 = var_63.value



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test the func_ann method of the Parser class.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'id'
    var_8 = 'MyClass'
    var_9 = 'int'
    var_10 = lambda r, node, self_ty='': var_8 if hasattr(node, var_7) and node.id == var_8 else var_9
    var_11 = 'self'
    var_12 = 'type[MyClass]'
    var_13 = lambda r, node, self_ty='': var_12 if hasattr(node, var_7) and node.id == var_8 else var_9
    var_14 = 'kwarg'



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Tests the `api` method of the Parser class by simulating the parsing \n    of a FunctionDef and a ClassDef, verifying that the internal state \n    (doc, docstring, level, root) is updated correctly.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_6 = 'x'
    var_7 = None
    var_8 = module_1.arg()
    var_9 = 'int'



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 'Self'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = module_1.arg()
    var_7 = 'x'
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.arg()
    var_12 = [var_6, var_11]
    var_13 = True
    var_14 = False
    var_15 = var_0.func_ann(var_1, var_12, has_self=var_13, cls_method=var_14)
    var_16 = list(var_15)
    var_17 = 'cls'
    var_18 = 'type'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = module_1.Load()
    var_22 = module_1.Name()
    var_23 = module_1.Subscript()
    var_24 = module_1.arg()
    var_25 = module_1.Load()
    var_26 = module_1.Name()
    var_27 = module_1.arg()
    var_28 = [var_24, var_27]
    var_29 = 'type[Self]'
    var_30 = var_0.func_ann(var_1, var_28, has_self=var_13, cls_method=var_13)
    var_31 = list(var_30)
    var_32 = 'args'
    var_33 = None
    var_34 = module_1.arg()
    var_35 = 'kwargs'
    var_36 = module_1.arg()
    var_37 = [var_34, var_36]
    var_38 = '*'
    var_39 = module_1.arg()
    var_40 = module_1.arg()
    var_41 = [var_39, var_40]
    var_42 = var_0.func_ann(var_1, var_41, has_self=var_14, cls_method=var_14)
    var_43 = list(var_42)
    var_44 = 'a'
    var_45 = module_1.arg()
    var_46 = 'b'
    var_47 = module_1.arg()
    var_48 = [var_45, var_47]
    var_49 = var_0.func_ann(var_1, var_48, has_self=var_14, cls_method=var_14)
    var_50 = list(var_49)
    var_51 = module_1.Load()
    var_52 = module_1.Name()
    var_53 = module_1.arg()
    var_54 = [var_53]
    var_55 = var_0.func_ann(var_1, var_54, has_self=var_14, cls_method=var_14)
    var_56 = list(var_55)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "\n    Tests the 'globals' method of the Parser class for various scenarios:\n    1. AnnAssign with name and value (Type Alias/Annotation).\n    2. Assign with type comment (Constant/Type Comment).\n    3. __all__ handling for imports/exports.\n    4. Ignoring non-target assignments or invalid targets.\n    "
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'x'
    var_6 = module_1.Name()
    var_7 = 5
    var_8 = module_1.Constant()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = module_1.AnnAssign()
    var_13 = 'pkg'
    var_14 = var_3.globals(var_13, var_12)
    var_15 = 'Y'
    var_16 = 10
    var_17 = module_1.Constant()
    var_18 = 'str'
    var_19 = 'pkg'
    var_20 = 'Z'
    var_21 = True
    var_22 = module_1.Constant()
    var_23 = 'pkg'
    var_24 = '__all__'
    var_25 = 'func1'
    var_26 = module_1.Constant()
    var_27 = 'func2'
    var_28 = module_1.Constant()
    var_29 = [var_26, var_28]
    var_30 = module_1.Load()
    var_31 = module_1.Tuple()
    var_32 = 'obj'
    var_33 = module_1.Name()
    var_34 = 'attr'
    var_35 = module_1.Constant()
    var_36 = 'a'
    var_37 = module_1.Name()
    var_38 = [var_37]
    var_39 = module_1.Constant()
    var_40 = module_1.Assign()
    var_41 = var_3.globals(var_7, var_40)



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Tests the Parser.api method to ensure it correctly populates doc and docstring \n    dictionaries for FunctionDef, AsyncFunctionDef, and ClassDef nodes, \n    and handles decorators and arguments appropriately.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = '# Module `pkg`'
    var_6 = set()
    var_7 = 'my_func'
    var_8 = 'x'
    var_9 = 'y'
    var_10 = [var_8, var_9]
    var_11 = 'int'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = 'pkg.my_func'
    var_15 = 'async_func'
    var_16 = 'z'
    var_17 = [var_16]
    var_18 = []
    var_19 = module_1.arg()
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = None
    var_25 = module_1.arguments(*var_20)
    var_26 = []
    var_27 = []
    var_28 = module_1.AsyncFunctionDef(*var_25)
    var_29 = var_3.api(var_4, var_28)
    var_30 = 'MyClass'
    var_31 = []
    var_32 = 'ATTR'
    var_33 = module_1.Constant()
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = []
    var_37 = 'deprecated'
    var_38 = module_1.Load()
    var_39 = module_1.Name()
    var_40 = 'dec_func'
    var_41 = 'a'
    var_42 = [var_41]
    var_43 = [var_39]
    var_44 = 'BaseClass'
    var_45 = module_1.Load()
    var_46 = module_1.Name()
    var_47 = 'SubClass'
    var_48 = [var_46]
    var_49 = []
    var_50 = []
    var_51 = module_1.ClassDef()
    var_52 = var_3.api(var_4, var_51)
    var_53 = 'nested_fn'
    var_54 = 'p'
    var_55 = [var_54]
    var_56 = 'Parent'
    var_57 = []
    var_58 = []



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Test the load_docstring method of the Parser class.\n    It should iterate through parsed names, check if they belong to the root module,\n    and extract docstrings from a provided module object using getdoc and doctest.\n    '
    var_1 = 'my_package'
    var_2 = True
    var_3 = False
    var_4 = module_0.Parser(var_2, toc=var_3, level=var_2)
    var_5 = 'my_package.sub_module'
    var_6 = 'other_package.module'
    var_7 = 'Content for sub'
    var_8 = 'Content for other'
    var_9 = 'sub_module'
    var_10 = 'other_package'
    var_11 = var_10 not in var_9
    var_12 = 'my_package.empty'
    var_13 = 'No doc here'



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test the is_public method of the Parser class.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.mod'
    var_6 = 'pkg.sub'
    var_7 = {var_6}
    var_8 = '# Module pkg'
    var_9 = '# Module sub'
    var_10 = '__main__'
    var_11 = 'is_public_family'
    var_12 = True
    var_13 = lambda x: var_12
    var_14 = 'pkg'
    var_15 = var_3.is_public(var_14)
    assert var_15 is True
    var_16 = {var_13}
    var_17 = 'all_sub'
    var_18 = '# sub'
    var_19 = 'is_public_family'
    var_20 = True
    var_21 = lambda x: var_20
    var_22 = 'pkg.sub'
    var_23 = var_3.is_public(var_22)
    assert var_23 is False
    var_24 = 'pkg.module'
    var_25 = {var_24}
    var_26 = '# pkg'
    var_27 = '# mod'
    var_28 = 'is_public_family'
    var_29 = True
    var_30 = lambda x: var_29
    var_31 = 'pkg.module'
    var_32 = var_3.is_public(var_31)
    assert var_32 is True
    var_33 = 'is_public_family'
    var_34 = False
    var_35 = lambda x: var_34
    var_36 = 'pkg.private'
    var_37 = var_3.is_public(var_36)
    assert var_37 is False
    var_38 = {var_37}
    var_39 = 'pkg.sub.child'
    var_40 = '# child'
    var_41 = 'int'
    var_42 = 'is_public_family'
    var_43 = True
    var_44 = lambda x: var_43
    var_45 = 'pkg.sub'
    var_46 = var_3.is_public(var_45)
    assert var_46 is True



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test the func_api method of Parser class.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = None
    var_8 = module_1.Constant()
    var_9 = 'pkg.my_function'
    var_10 = 'int'
    var_11 = 'str'
    var_12 = [var_10, var_11]
    var_13 = 'pkg'
    var_14 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api specifically for classmethods (Self handling).'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'type'
    var_6 = module_1.Load()
    var_7 = None
    var_8 = 'pkg.my_class_method'
    var_9 = 'type[Self]'
    var_10 = [var_9]
    var_11 = 'pkg'
    var_12 = None
    var_13 = True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Tests the func_ann method of Parser class.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'self'
    var_6 = 'MyClass'
    var_7 = module_1.Load()
    var_8 = 'x'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = 'cls'
    var_12 = 'Type[MyClass]'
    var_13 = module_1.Load()
    var_14 = 'y'
    var_15 = 'str'
    var_16 = module_1.Load()
    var_17 = 'a'
    var_18 = 'args'
    var_19 = None
    var_20 = 'kwargs'
    var_21 = module_1.Load()
    var_22 = 'unannotated'
    var_23 = '<ANY>'
    var_24 = 'Any'
    var_25 = '*'
    var_26 = 'b'
    var_27 = module_1.Load()



# Parsed testcases at query #2
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the class_api method of the Parser class to ensure it correctly \n    processes bases, enums, and members (public attributes) from a class body.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg.mod'
    var_5 = 'pkg.mod.MyClass'
    var_6 = 2
    var_7 = 'id'
    var_8 = 'public_attr'
    var_9 = 'int'
    var_10 = 'str'
    var_11 = lambda root, node: var_9 if hasattr(node, var_7) and node.id == var_8 else var_10
    var_12 = 'pkg.mod'
    var_13 = 'pkg.mod.MyClass'
    var_14 = []
    var_15 = var_3.class_api(var_12, var_13, var_14, var_4)
    var_16 = 'enum.Base'
    var_17 = 'pkg.mod'
    var_18 = 'pkg.mod.MyEnum'
    var_19 = var_3.class_api(var_17, var_18, var_4, var_15)
    var_20 = 'Existing'
    var_21 = 'pkg.mod.MyClass.OLD'
    var_22 = 'pkg.mod'
    var_23 = 'pkg.mod.MyClass'
    var_24 = []
    var_25 = []
    var_26 = var_3.class_api(var_22, var_23, var_24, var_25)



# Parsed testcases at query #3
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Test the globals method of the Parser class for various scenarios:\n    1. AnnAssign with a Name target (Type Alias / Annotation).\n    2. Assign with a single Name target and type comment (Constant via type comment).\n    3. Assign with a single Name target and no type comment (Constant via const_type).\n    4. __all__ handling (Updating self.imp).\n    5. Non-target assignments or complex assignments (Should be ignored).\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'x: int = 10'
    var_6 = 0
    var_7 = module_1.parse(var_5)
    var_8 = var_7.body[var_6]
    var_9 = 'pkg'
    var_10 = var_3.globals(var_9, var_8)
    var_11 = 'Y = 20'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_6]
    var_14 = var_3.globals(var_9, var_13)
    var_15 = 'Z = 30'
    var_16 = module_1.parse(var_15)
    var_17 = var_16.body[var_6]
    var_18 = var_3.globals(var_9, var_17)
    var_19 = "__all__ = ('a', 'b')"
    var_20 = module_1.parse(var_19)
    var_21 = var_20.body[var_6]
    var_22 = var_3.globals(var_9, var_21)
    var_23 = 'x, y = 1, 2'
    var_24 = module_1.parse(var_23)
    var_25 = var_24.body[var_6]
    var_26 = var_3.globals(var_9, var_25)
    var_27 = '[x] = [1]'
    var_28 = module_1.parse(var_27)
    var_29 = var_28.body[var_9]
    var_30 = var_3.globals(var_10, var_29)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test edge cases like empty values or non-string constants.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'x: int = None'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_2]
    var_8 = None
    var_9 = 'pkg'
    var_10 = var_3.globals(var_9, var_7)
    var_11 = 'small_val = 5'
    var_12 = module_1.parse(var_11)
    var_13 = var_12.body[var_2]
    var_14 = 'pkg'
    var_15 = var_3.globals(var_14, var_13)



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
    var_8 = var_4[var_6]
    var_9 = var_8.targets[var_6]
    var_10 = 'if True: x = 1\nelse: y = 2'
    var_11 = module_0.parse(var_10)
    var_12 = var_11.body
    var_13 = module_1.walk_body(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = '\ntry:\n    x = 1\nexcept ValueError:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n'
    var_17 = module_0.parse(var_16)
    var_18 = var_17.body
    var_19 = module_1.walk_body(var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = '\ntry:\n    if True:\n        a = 1\nexcept:\n    pass\n'
    var_23 = module_0.parse(var_22)
    var_24 = var_23.body
    var_25 = module_1.walk_body(var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = var_26[var_6]
    var_29 = var_26[var_6]
    var_30 = var_29.targets[var_6]
    var_31 = ''
    var_32 = module_0.parse(var_31)
    var_33 = var_32.body
    var_34 = module_1.walk_body(var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 0



# Parsed testcases at query #5
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mylib'
    var_1 = 'mylib.MyType'
    var_2 = 'mylib.ComplexType'
    var_3 = 'mylib.TypeVarRef'
    var_4 = 'int'
    var_5 = 'str | int'
    var_6 = "typing.TypeVar('T')"
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'SelfType'
    var_9 = module_0.Resolver(var_0, var_7, var_8)
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = var_9.visit_Name(var_11)
    var_13 = module_0.Resolver(var_0, var_7)
    var_14 = 'MyType'
    var_15 = module_1.Load()
    var_16 = module_1.Name()
    var_17 = var_13.visit_Name(var_16)
    var_18 = 'ComplexType'
    var_19 = module_1.Load()
    var_20 = module_1.Name()
    var_21 = var_13.visit_Name(var_20)
    var_22 = 'TypeVarRef'
    var_23 = module_1.Load()
    var_24 = module_1.Name()
    var_25 = var_13.visit_Name(var_24)
    var_26 = 'Unknown'
    var_27 = module_1.Load()
    var_28 = module_1.Name()
    var_29 = 'pkg'
    var_30 = 'pkg.Sub'
    var_31 = {var_30: var_4}
    var_32 = module_0.Resolver(var_29, var_31)
    var_33 = 'Sub'
    var_34 = module_1.Load()
    var_35 = module_1.Name()
    var_36 = var_32.visit_Name(var_35)



# Parsed testcases at query #6
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    The test verifies that the method correctly:\n    1. Performs alias substitution via __find_alias.\n    2. Generates a Table of Contents if toc is True.\n    3. Formats and joins documentation entries for public members.\n    4. Incorporates constants and docstrings.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.mod'
    var_5 = 0
    var_6 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_7 = '## mod\n<a id="pkg.mod"></a>\n\n*Full name:* `pkg.mod`\n\n'
    var_8 = '\nDocstring for mod.'
    var_9 = 'pkg.CONST'
    var_10 = 'int'
    var_11 = set()
    var_12 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Tests compile method without Table of Contents.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = '# Module `pkg`\n\n'
    var_6 = set()
    var_7 = var_3.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Tests that private members (not public) are filtered out during compilation.'
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg._private'
    var_5 = 0
    var_6 = '# Module `pkg`\n\n'
    var_7 = '## private\n\n'
    var_8 = set()
    var_9 = var_2.compile()



# Parsed testcases at query #7
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Test the compile method of the Parser class.\n    This test verifies that the compile method correctly:\n    1. Performs alias substitution via __find_alias.\n    2. Generates a Table of Contents (TOC) if toc is True.\n    3. Formats documentation strings with links and IDs.\n    4. Includes constants from the const dictionary.\n    5. Integrates docstrings.\n    6. Sorts entries based on the internal __names_cmp logic.\n    '
    var_1 = True
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg.mod'
    var_4 = 'pkg.mod.func'
    var_5 = 2
    var_6 = '# Module `pkg.mod`\n<a id="pkg-mod"></a>\n\n'
    var_7 = '## func()\n\n*Full name:* `pkg.mod.func`\n<a id="pkg-mod-func"></a>\n\n'
    var_8 = 'Module docstring.'
    var_9 = 'Function docstring.'
    var_10 = 'pkg.mod.CONST'
    var_11 = 'int'
    var_12 = set()
    var_13 = var_2.compile()
    var_14 = var_2.compile()
    var_15 = 'pkg.mod.private'
    var_16 = '# Module `pkg.mod`\n\n'
    var_17 = '## private()\n\n'
    var_18 = var_2.compile()



# Parsed testcases at query #8
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "\n    Tests the 'imports' method of the Parser class for various Import and ImportFrom scenarios.\n    "
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'os'
    var_5 = None
    var_6 = [var_2]
    var_7 = module_1.Import()
    var_8 = 'my_pkg'
    var_9 = var_3.imports(var_8, var_7)
    var_10 = 'pandas'
    var_11 = 'pd'
    var_12 = 'math'
    var_13 = 'sqrt'
    var_14 = 0
    var_15 = 'utils'
    var_16 = 'helper'
    var_17 = None
    var_18 = [var_6]
    var_19 = 1
    var_20 = module_1.ImportFrom()
    var_21 = 'my_pkg'
    var_22 = var_3.imports(var_21, var_20)
    var_23 = 'utils.helper'
    var_24 = 'sub'
    var_25 = 'func'
    var_26 = None
    var_27 = [var_6]
    var_28 = 2
    var_29 = module_1.ImportFrom()
    var_30 = 'my_pkg'
    var_31 = var_3.imports(var_30, var_29)
    var_32 = var_3.alias
    var_33 = len(var_32)
    var_34 = var_3.alias



# Parsed testcases at query #9
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = "\n    Test the func_api method of the Parser class.\n    This test verifies that function metadata (arguments, returns, defaults) \n    is correctly processed and appended to the parser's documentation dictionary.\n    "
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.func'
    var_6 = 'a'
    var_7 = None
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = 'b'
    var_11 = 'int'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = module_1.arg()
    var_15 = 'c'
    var_16 = 'str'
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = module_1.arg()
    var_20 = [var_14, var_19]
    var_21 = 'val'
    var_22 = module_1.Constant()
    var_23 = [var_22]
    var_24 = 'args'
    var_25 = module_1.arg()
    var_26 = []
    var_27 = []
    var_28 = 'kwargs'
    var_29 = module_1.arg()
    var_30 = module_1.arguments(*var_20)
    var_31 = 'bool'
    var_32 = module_1.Load()
    var_33 = module_1.Name()
    var_34 = 'pkg'
    var_35 = False
    var_36 = var_3.func_api(var_34, var_5, var_30, var_33, has_self=var_35, cls_method=var_35)
    var_37 = False
    var_38 = True
    var_39 = []
    var_40 = 'x'
    var_41 = module_1.arg()
    var_42 = [var_41]
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = module_1.arguments(*var_42)
    var_47 = 'pkg'
    var_48 = 'pkg.method'
    var_49 = None
    var_50 = True
    var_51 = False
    var_52 = var_3.func_api(var_47, var_48, var_46, var_49, has_self=var_50, cls_method=var_51)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test the func_ann generator specifically.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = []
    var_5 = 'a'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.arg()
    var_10 = [var_9]
    var_11 = 2
    var_12 = module_1.Constant()
    var_13 = [var_12]
    var_14 = None
    var_15 = []
    var_16 = []
    var_17 = module_1.arguments(*var_10)
    var_18 = 'pkg'
    var_19 = False
    var_20 = var_3.func_ann(var_18, var_17, has_self=var_19, cls_method=var_19)
    var_21 = list(var_20)



# Parsed testcases at query #10
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = '\n    Test the func_api method of the Parser class.\n    This test verifies that function annotations, arguments, \n    defaults, and return types are correctly processed into the doc dictionary.\n    '
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'pkg.my_function'
    var_6 = 10
    var_7 = module_1.Constant()
    var_8 = 'pkg'
    var_9 = None
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test func_api specifically with self/cls_method parameters.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = 'Self'
    var_6 = module_1.Load()
    var_7 = 'pkg'
    var_8 = 'pkg.method'
    var_9 = None
    var_10 = True
    var_11 = False



# Parsed testcases at query #11
#--------------------------


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mod'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 123
    var_5 = module_1.Constant()
    var_6 = var_3.visit_Constant(var_5)
    var_7 = {}
    var_8 = module_0.Resolver(var_0, var_7, var_2)
    var_9 = "'unclosed string"
    var_10 = module_1.Constant()
    var_11 = var_8.visit_Constant(var_10)
    var_12 = 'mod.test_name'
    var_13 = 'test_name'
    var_14 = {var_12: var_13}
    var_15 = module_0.Resolver(var_0, var_14, var_2)
    var_16 = module_1.Constant()
    var_17 = var_15.visit_Constant(var_16)
    var_18 = {}
    var_19 = module_0.Resolver(var_0, var_18, var_2)
    var_20 = '1 + 1'
    var_21 = module_1.Constant()
    var_22 = var_19.visit_Constant(var_21)



# Parsed testcases at query #12
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    var_2 = 'sys.path'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'my_module.sub_module.Class'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = 'module.__init__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = 'package.module.__doc__'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is True
    var_10 = '_private'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is False
    var_12 = 'package._module'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is False
    var_14 = 'package.module._Class'
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is False
    var_16 = '__builtin__'
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = 'package.__private__.module'
    var_19 = module_0.is_public_family(var_18)
    assert var_19 is False
    var_20 = ''
    var_21 = module_0.is_public_family(var_20)
    assert var_21 is True
    var_22 = '.'
    var_23 = module_0.is_public_family(var_22)
    assert var_23 is True



# Parsed testcases at query #13
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''
    var_2 = 'This is a simple description.'
    var_3 = module_0.doctest(var_2)
    var_4 = '>>> 1 + 1\n2'
    var_5 = '```python\n>>> 1 + 1\n2\n```'
    var_6 = module_0.doctest(var_4)
    var_7 = '>>> add(1, 2)\n3\nThis is a description of the function.'
    var_8 = '```python\n>>> add(1, 2)\n3\n```\nThis is a description of the function.'
    var_9 = module_0.doctest(var_7)
    var_10 = 'Check this:\n>>> len([1, 2])\n2'
    var_11 = 'Check this:\n```python\n>>> len([1, 2])\n2\n```'
    var_12 = module_0.doctest(var_10)
    var_13 = ">>> print('hi')\nhi\nNext line."
    var_14 = "```python\n>>> print('hi')\nhi\n```\nNext line."
    var_15 = module_0.doctest(var_13)
    var_16 = '>>> x = 1\nx\nIntermediate text\n>>> y = 2\ny'
    var_17 = '```python\n>>> x = 1\nx\n```\nIntermediate text\n```python\n>>> y = 2\ny\n```'
    var_18 = module_0.doctest(var_16)



# Parsed testcases at query #14
#--------------------------


import apimd.parser as module_0

def test_case_0():
    var_0 = '\n    Tests the compile method of the Parser class.\n    Since compile() relies on a complex internal state (alias substitution, \n    doc generation, and sorting), we mock the internal dependencies \n    to verify the string construction logic.\n    '
    var_1 = True
    assert var_1 == 2
    var_2 = module_0.Parser(var_1, toc=var_1, level=var_1)
    var_3 = 'pkg'
    var_4 = 'pkg.mod'
    var_5 = 0
    var_6 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_7 = '## mod()\n\n*Full name:* `pkg.mod`\n\n'
    var_8 = 'Doc for mod.'
    var_9 = set()
    var_10 = var_2.compile()
    var_11 = '**Table of contents:**'
    var_12 = 1
    var_13 = '+ [pkg](#pkg)'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Tests compile method without Table of Contents.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, toc=var_2, level=var_1)
    var_4 = 'pkg'
    var_5 = '# Module `pkg`\n\n'
    var_6 = set()



# Parsed testcases at query #15
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = 'Test behavior when parsing fails (though ast.parse usually raises before method body).'
    var_1 = 'pkg'
    var_2 = 'invalid code'
    var_3 = module_0.parse(var_1, var_2)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Tests the is_public method of the Parser class.'
    var_1 = 'pkg'
    var_2 = 'pkg.Module'
    var_3 = 'doc'
    var_4 = {var_2: var_3}
    var_5 = 'pkg._Private'
    var_6 = {var_5: var_3}
    var_7 = 'pkg.func'
    var_8 = {var_7}
    var_9 = {var_1: var_8}
    var_10 = 'pkg.sub'
    var_11 = {var_10}
    var_12 = {var_1: var_11}
    var_13 = 'pkg.other'
    var_14 = {var_13}
    var_15 = {var_1: var_14}
    var_16 = {var_13: var_3}
    var_17 = {var_10}
    var_18 = {var_1: var_17}
    var_19 = 'pkg.sub.Child'
    var_20 = {var_19: var_3}
    var_21 = 'pkg.CONST'
    var_22 = 'val'
    var_23 = {var_21: var_22}
    var_24 = {var_21: var_3}
    var_25 = 'pkg'
    var_26 = 'unknown.name'
    var_27 = {var_26}
    var_28 = {var_25: var_27}



