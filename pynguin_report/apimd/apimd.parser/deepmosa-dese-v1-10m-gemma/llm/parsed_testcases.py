####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parser_imports_import_node. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_import_node_asname. Retrieved 4/8 statements.
# Partially parsed test_parser_imports_import_from_absolute. Retrieved 6/10 statements.
# Partially parsed test_parser_imports_with_all_filter. Retrieved 10/15 statements.
# Partially parsed test_parser_imports_empty_names. Retrieved 4/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'sub'
    var_3 = 'func'
    var_4 = 'f'
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'other.lib'
    var_3 = 'ol'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = '__all__'
    var_3 = 'func_a'
    var_4 = module_1.Constant()
    var_5 = 'func_b'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'mod'
    var_3 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_public_logic_basic. Retrieved 7/9 statements.
# Partially parsed test_is_public_with_all_export. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_submodule_in_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_via_import_chain. Retrieved 3/7 statements.
# Partially parsed test_is_public_with_constants. Retrieved 3/6 statements.
# Partially parsed test_is_public_with_private_submodule_in_all. Retrieved 3/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True
    var_3 = 'pkg._private'
    var_4 = var_0.is_public(var_3)
    assert var_4 is False
    var_5 = 'pkg.__init__'
    var_6 = var_0.is_public(var_5)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.exported'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True
    var_3 = 'pkg._hidden'
    var_4 = var_0.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True
    var_3 = 'pkg.sub.member'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.CONST'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg._private_mod'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_compile_empty. Retrieved 4/12 statements.
# Partially parsed test_parser_compile_with_toc. Retrieved 11/19 statements.
# Partially parsed test_parser_compile_with_content. Retrieved 12/20 statements.
# Partially parsed test_parser_compile_filtering_magic. Retrieved 11/19 statements.
# Partially parsed test_parser_compile_is_public_filter. Retrieved 11/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = True
    var_2 = ''
    var_3 = var_0.compile()
    assert var_3 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, toc=var_0, level=var_0)
    var_2 = 'pkg'
    var_3 = 'pkg.mod'
    var_4 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_5 = '## mod'
    var_6 = 0
    var_7 = set()
    var_8 = ''
    var_9 = '**Table of contents:**\n+ [`pkg`](#pkg)\n    + [`pkg.mod`](#pkg-mod)\n\n# Module `pkg`\n<a id="pkg"></a>\n\n\n## mod\n'
    var_10 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg.mod'
    var_5 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_6 = '## mod\nContent'
    var_7 = set()
    var_8 = ''
    var_9 = 'Doc'
    var_10 = '# Module `pkg`\n<a id="pkg"></a>\n\n\n## mod\nContentDoc'
    var_11 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg.__init__'
    var_5 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_6 = '## init'
    var_7 = set()
    var_8 = ''
    var_9 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_10 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg._private'
    var_5 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_6 = '## private'
    var_7 = set()
    var_8 = ''
    var_9 = '# Module `pkg`\n<a id="pkg"></a>\n\n'
    var_10 = var_2.compile()



# Parsed testcases at query #4
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Constant()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'a'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1.0
    var_1 = module_0.Constant()
    var_2 = 2.5
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[float]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Constant()
    var_2 = False
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Set()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'set[bool]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = 1
    var_6 = module_0.Constant()
    var_7 = 2
    var_8 = module_0.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_0.Dict()
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'dict[str, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 1
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Constant()
    var_6 = [var_5]
    var_7 = module_0.Dict()
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'dict[Any, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = []
    var_4 = module_0.Call(*var_3)
    var_5 = module_1.const_type(var_4)
    assert var_5 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = 'str'
    var_4 = module_0.Load()
    var_5 = module_0.Attribute()
    var_6 = []
    var_7 = module_0.Call(*var_6)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.List()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'list[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Tuple()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'tuple[]'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parser_parse_module_level_docstring. Retrieved 6/11 statements.
# Partially parsed test_parser_parse_imports. Retrieved 8/15 statements.
# Partially parsed test_parser_parse_assignment. Retrieved 12/19 statements.
# Partially parsed test_parser_parse_docstring_assignment. Retrieved 6/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'my_pkg.module'
    var_4 = "print('hello')"
    var_5 = var_2.parse(var_3, var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = module_0.Parser(var_2, toc=var_3, level=var_2)
    var_5 = 'my_pkg'
    var_6 = 'import os'
    var_7 = var_4.parse(var_5, var_6)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'X'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'int'
    var_4 = module_0.Load()
    var_5 = module_0.Name()
    var_6 = True
    var_7 = False
    var_8 = module_1.Parser(var_6, toc=var_7, level=var_6)
    var_9 = 'my_pkg'
    var_10 = 'X: int = 1'
    var_11 = var_8.parse(var_9, var_10)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'my_pkg'
    var_4 = ''
    var_5 = var_2.parse(var_3, var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_imports_not_isinstance_Import. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 0
    var_5 = 'pkg'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_visit_Subscript_union_to_bitor. Retrieved 22/30 statements.
# Partially parsed test_visit_Subscript_optional_to_or_none. Retrieved 16/23 statements.
# Partially parsed test_visit_Subscript_pep585_replacement. Retrieved 16/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = [var_10, var_13]
    var_15 = module_1.Load()
    var_16 = module_1.Tuple()
    var_17 = module_1.Load()
    var_18 = module_1.Subscript()
    var_19 = var_4.visit_Subscript(var_18)
    var_20 = var_19.op
    var_21 = var_19.left

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Optional'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.Load()
    var_12 = module_1.Subscript()
    var_13 = var_4.visit_Subscript(var_12)
    var_14 = var_13.op
    var_15 = var_13.right

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'PEP585'
    var_1 = {}
    var_2 = 'pkg.List'
    var_3 = 'list'
    var_4 = 'pkg'
    var_5 = {}
    var_6 = module_0.Resolver(var_4, var_5)
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = module_1.Load()
    var_14 = module_1.Subscript()
    var_15 = var_6.visit_Subscript(var_14)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.Dict'
    var_2 = 'typing.Dict'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Dict'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = [var_10]
    var_12 = module_1.Load()
    var_13 = module_1.Tuple()
    var_14 = module_1.Load()
    var_15 = module_1.Subscript()
    var_16 = var_4.visit_Subscript(var_15)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = module_1.Load()
    var_13 = module_1.Subscript()
    var_14 = var_2.visit_Subscript(var_13)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parser_api_function_def. Retrieved 5/25 statements.
# Partially parsed test_parser_api_class_def. Retrieved 5/18 statements.
# Partially parsed test_parser_api_async_function_def. Retrieved 5/25 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'pkg'



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 'test'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = (var_0, var_0)
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a|b'
    var_1 = 'x&y'
    var_2 = (var_0, var_1)
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parser_class_api_with_members. Retrieved 25/40 statements.
# Partially parsed test_parser_class_api_with_enum_bases. Retrieved 26/35 statements.
# Partially parsed test_parser_class_api_with_bases. Retrieved 18/23 statements.
# Partially parsed test_parser_class_api_with_deletion. Retrieved 17/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'MyClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.ClassDef()
    var_9 = 'PUBLIC_MEMBER'
    var_10 = 42
    var_11 = module_1.Constant()
    var_12 = None
    var_13 = 'TYPED_MEMBER'
    var_14 = 'hello'
    var_15 = module_1.Constant()
    var_16 = 'str'
    var_17 = module_1.Load()
    var_18 = module_1.Name()
    var_19 = 'int'
    var_20 = 'pkg.MyClass'
    var_21 = []
    var_22 = var_8.body
    var_23 = var_2.class_api(var_3, var_20, var_21, var_22)
    var_24 = var_2.doc[var_20]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'MyEnum'
    var_5 = 'enum'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'Enum'
    var_9 = module_1.Load()
    var_10 = module_1.Attribute()
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = module_1.ClassDef()
    var_15 = 'RED'
    var_16 = module_1.Constant()
    var_17 = None
    var_18 = 'pkg.MyEnum'
    var_19 = 'enum.Enum'
    var_20 = module_1.Load()
    var_21 = module_1.Name()
    var_22 = [var_21]
    var_23 = var_14.body
    var_24 = var_2.class_api(var_3, var_18, var_22, var_23)
    var_25 = var_2.doc[var_18]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'BaseClass'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'SubClass'
    var_8 = [var_6]
    var_9 = []
    var_10 = []
    var_11 = module_1.ClassDef()
    var_12 = 'pkg.BaseClass'
    var_13 = 'pkg.SubClass'
    var_14 = [var_6]
    var_15 = []
    var_16 = var_2.class_api(var_3, var_13, var_14, var_15)
    var_17 = var_2.doc[var_13]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'pkg'
    var_4 = 'TEMP'
    var_5 = module_1.Constant()
    var_6 = None
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = module_1.Delete()
    var_11 = 'MyClass'
    var_12 = []
    var_13 = []
    var_14 = 'pkg.MyClass'
    var_15 = []
    var_16 = var_2.doc[var_14]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_parser_globals_assignment_with_type_comment. Retrieved 7/23 statements.
# Partially parsed test_parser_globals_assignment_without_type_comment_infers_constant. Retrieved 6/19 statements.
# Partially parsed test_parser_globals_annassign. Retrieved 9/21 statements.
# Partially parsed test_parser_globals_ignores_non_uppercase_for_const. Retrieved 6/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONSTANT'
    var_3 = 'pkg.MY_CONSTANT'
    var_4 = 10
    var_5 = module_1.Constant()
    var_6 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'VERSION'
    var_3 = 'pkg.VERSION'
    var_4 = '1.0'
    var_5 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'VAL'
    var_3 = 'pkg.VAL'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 5
    var_8 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'not_a_constant'
    var_3 = 'pkg.not_a_constant'
    var_4 = 1
    var_5 = module_1.Constant()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_compile_skips_magic_names_without_docstrings. Retrieved 2/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_class_api_mem_not_empty. Retrieved 6/43 statements.


import ast as module_0

def test_case_0():
    var_0 = 'PUBLIC_ATTR'
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = 'pkg'
    var_4 = 'pkg.MyClass'
    var_5 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 10/12 statements.
# Partially parsed test_visit_Attribute_leaves_non_typing_attribute_unchanged. Retrieved 10/12 statements.
# Partially parsed test_visit_Attribute_leaves_complex_attribute_value_unchanged. Retrieved 14/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'collections'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'deque'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'a'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'b'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'c'
    var_10 = module_1.Load()
    var_11 = module_1.Attribute()
    var_12 = var_2.visit_Attribute(var_11)
    var_13 = var_12.value



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_annassign_is_public. Retrieved 5/45 statements.


def test_case_0():
    var_0 = 'PUBLIC_ATTR'
    var_1 = []
    var_2 = False
    var_3 = {}
    var_4 = 'root'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_api_async_function_doc_generation. Retrieved 15/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1)
    var_3 = 'async_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = module_1.arguments(*var_6)
    var_12 = module_1.AsyncFunctionDef(*var_11)
    var_13 = 'pkg'
    var_14 = var_2.api(var_13, var_12)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_globals_predicate_false_when_not_upper. Retrieved 6/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'x'
    var_4 = module_1.Constant()
    var_5 = 'pkg'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_api_async_function_doc_generation. Retrieved 14/46 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = module_1.AsyncFunctionDef(*var_8)
    var_11 = 'decorator'
    var_12 = 'pkg'
    var_13 = var_0.api(var_12, var_10)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_compile_skips_magic_names_without_docstrings. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = var_0.compile()
    var_3 = '.'
    var_4 = '-'
    var_5 = f'+ [{var_1}](#{var_0.replace(var_3, var_4)})'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_parser_globals_ann_assign_with_type_comment. Retrieved 9/16 statements.
# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_assign_without_type_comment_inference. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_all_updates_imports. Retrieved 11/20 statements.
# Partially parsed test_parser_globals_ignores_non_name_targets. Retrieved 9/17 statements.
# Partially parsed test_parser_globals_ignores_non_assign_nodes. Retrieved 8/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = '__all__'
    var_3 = 'A'
    var_4 = module_1.Constant()
    var_5 = 'B'
    var_6 = module_1.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_1.Load()
    var_9 = module_1.Tuple()
    var_10 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'obj'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'ATTR'
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 1
    var_3 = module_1.Constant()
    var_4 = module_1.Expr()
    var_5 = var_0.globals(var_1, var_4)
    var_6 = var_0.alias
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "print('hello')"

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1 + 1\n2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> 1 + 1\n2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 5\n>>> print(x)\n5'
    var_1 = '```python\n>>> x = 5\n>>> print(x)\n5\n```'
    var_2 = module_0.doctest(var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Intro text\n>>> 1\n2\nOutro text'
    var_1 = 'Intro text\n```python\n>>> 1\n2\n```\nOutro text'
    var_2 = module_0.doctest(var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> 1\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> 1\n2\n'
    var_1 = '```python\n>>> 1\n2\n```'
    var_2 = module_0.doctest(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #4
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = 'x = 10\nclass MyClass:\n    pass'
    var_4 = 'my_module'
    var_5 = var_2.parse(var_4, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 6/12 statements.
# Partially parsed test_parser_globals_assign_without_type_comment. Retrieved 6/12 statements.
# Partially parsed test_parser_globals_annassign. Retrieved 9/17 statements.
# Partially parsed test_parser_globals_all_import_logic. Retrieved 12/19 statements.
# Partially parsed test_parser_globals_ignores_non_target_names. Retrieved 11/17 statements.
# Partially parsed test_parser_globals_ignores_non_single_target_assign. Retrieved 7/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'OTHER_CONST'
    var_3 = 'hello'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'float'
    var_3 = 'VAL'
    var_4 = 1.5
    var_5 = module_1.Constant()
    var_6 = 'f'
    var_7 = module_1.Load()
    var_8 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'sub_mod'
    var_5 = module_1.Constant()
    var_6 = 'other'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'obj'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'ATTR'
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = None
    var_9 = var_0.alias
    var_10 = len(var_9)
    assert var_10 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parser_api_function_def. Retrieved 24/30 statements.
# Partially parsed test_parser_api_class_def. Retrieved 17/29 statements.
# Partially parsed test_parser_api_with_decorators. Retrieved 19/25 statements.
# Partially parsed test_parser_api_async_function. Retrieved 16/22 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 0
    var_3 = 'pkg.func'
    var_4 = 'func'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.arg()
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 10
    var_15 = module_1.Constant()
    var_16 = [var_15]
    var_17 = None
    var_18 = module_1.arguments(*var_11)
    var_19 = []
    var_20 = []
    var_21 = module_1.FunctionDef(*var_18)
    var_22 = 'pkg'
    var_23 = var_0.api(var_22, var_21)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 0
    var_3 = 'MyClass'
    var_4 = 'Base'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_6]
    var_8 = []
    var_9 = 'ATTR'
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = 5
    var_14 = module_1.Constant()
    var_15 = []
    var_16 = 'pkg'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 0
    var_3 = 'deco'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'func'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = module_1.arguments(*var_8)
    var_14 = []
    var_15 = [var_5]
    var_16 = module_1.FunctionDef(*var_13)
    var_17 = 'pkg'
    var_18 = var_0.api(var_17, var_16)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 0
    var_3 = 'async_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = module_1.arguments(*var_5)
    var_11 = []
    var_12 = []
    var_13 = module_1.AsyncFunctionDef(*var_10)
    var_14 = 'pkg'
    var_15 = var_0.api(var_14, var_13)



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nfrom sys import argv'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_func_ann_self_with_annotation. Retrieved 8/16 statements.
# Partially parsed test_func_ann_cls_method_with_type_wrapper. Retrieved 7/17 statements.
# Partially parsed test_func_ann_basic_logic. Retrieved 20/47 statements.
# Partially parsed test_func_ann_complex_sequence. Retrieved 14/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'MyClass'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'pkg'
    var_6 = True
    var_7 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'type[MyClass]'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'pkg'
    var_6 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'pkg'
    var_5 = False
    var_6 = 'int'
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = 2
    var_10 = module_1.Constant()
    var_11 = '*'
    var_12 = 'self'
    var_13 = 'S'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = True
    var_17 = 'type[S]'
    var_18 = True
    var_19 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'T'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = '*'
    var_6 = None
    var_7 = 'k'
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = 'int'
    var_11 = 'pkg'
    var_12 = True
    var_13 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = []
    var_3 = False
    var_4 = var_0.func_ann(var_1, var_2, has_self=var_3, cls_method=var_3)
    var_5 = list(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_parser_globals_assign_with_type_comment. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_assign_without_type_comment. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_annassign. Retrieved 8/17 statements.
# Partially parsed test_parser_globals_all_list_updates_imports. Retrieved 12/21 statements.
# Partially parsed test_parser_globals_ignores_non_upper_case_for_const. Retrieved 6/14 statements.
# Partially parsed test_parser_globals_skips_unsupported_nodes. Retrieved 10/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'MY_CONST'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'OTHER_CONST'
    var_3 = 'hello'
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'int'
    var_3 = 'ANNO_VAR'
    var_4 = 5
    var_5 = module_1.Constant()
    var_6 = module_1.Load()
    var_7 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'MOD_A'
    var_5 = module_1.Constant()
    var_6 = 'MOD_B'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Load()
    var_10 = module_1.Tuple()
    var_11 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'lowercase_var'
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 1
    var_3 = module_1.Constant()
    var_4 = module_1.Expr()
    var_5 = var_0.globals(var_1, var_4)
    var_6 = var_0.alias
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.const
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = True
    var_1 = 3
    var_2 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_1, toc=var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_level. Retrieved 2/7 statements.
# Partially parsed test_attr_non_existent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nested_non_existent_attribute. Retrieved 1/7 statements.
# Partially parsed test_attr_broken_chain_at_middle. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'value'

def test_case_0():
    var_0 = 42
    var_1 = 'child.value'

def test_case_0():
    var_0 = 'missing'

def test_case_0():
    var_0 = 'child.missing'

def test_case_0():
    var_0 = None
    var_1 = 'child.value'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'any'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parser_class_api_with_members. Retrieved 13/21 statements.
# Partially parsed test_parser_class_api_with_bases. Retrieved 13/17 statements.
# Partially parsed test_parser_class_api_with_enum_style. Retrieved 12/21 statements.
# Partially parsed test_parser_class_api_with_deletion. Retrieved 16/24 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = '## class MyClass\n\n*Full name:* `pkg.MyClass`\n\n'
    var_5 = 'MY_CONST'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 10
    var_10 = module_1.Constant()
    var_11 = 'MyClass'
    var_12 = []

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = 'pkg.Base'
    var_5 = 'Base'
    var_6 = '## class MyClass\n\n*Full name:* `pkg.MyClass`\n\n'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'MyClass'
    var_10 = [var_8]
    var_11 = []
    var_12 = var_0.class_api(var_2, var_9, var_10, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyEnum'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = '## class MyEnum\n\n*Full name:* `pkg.MyEnum`\n\n'
    var_5 = 'RED'
    var_6 = module_1.Constant()
    var_7 = 'enum.Enum'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'MyEnum'
    var_11 = [var_9]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.MyClass'
    var_2 = 'pkg'
    var_3 = 1
    var_4 = '## class MyClass\n\n*Full name:* `pkg.MyClass`\n\n'
    var_5 = 'OLD_MEMBER'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Constant()
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_11]
    var_13 = module_1.Delete()
    var_14 = 'MyClass'
    var_15 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_visit_Subscript_union_to_bitor. Retrieved 23/31 statements.
# Partially parsed test_visit_Subscript_optional_to_bitor_none. Retrieved 16/22 statements.
# Partially parsed test_visit_Subscript_pep585_conversion. Retrieved 18/25 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'mypkg.Union'
    var_2 = 'typing.Union'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Union'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 'str'
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = [var_10, var_13]
    var_15 = module_1.Load()
    var_16 = module_1.Tuple()
    var_17 = module_1.Load()
    var_18 = module_1.Subscript()
    var_19 = var_4.visit_Subscript(var_18)
    var_20 = var_19.op
    var_21 = var_19.left
    var_22 = var_19.right

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'mypkg.Optional'
    var_2 = 'typing.Optional'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Optional'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.Load()
    var_12 = module_1.Subscript()
    var_13 = var_4.visit_Subscript(var_12)
    var_14 = var_13.op
    var_15 = var_13.right

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'typing.List'
    var_1 = 'list'
    var_2 = {var_0: var_1}
    var_3 = 'mypkg'
    var_4 = 'mypexp.List'
    var_5 = {var_4: var_0}
    var_6 = module_0.Resolver(var_3, var_5)
    var_7 = 'List'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = module_1.Load()
    var_14 = module_1.Subscript()
    var_15 = 'mypkg.List'
    var_16 = var_6.visit_Subscript(var_14)
    var_17 = var_16.value

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = module_1.Load()
    var_13 = module_1.Subscript()
    var_14 = var_2.visit_Subscript(var_13)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'other.Type'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    var_4 = 'List'
    var_5 = module_1.Load()
    var_6 = str(var_5)
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = module_1.Load()
    var_12 = module_1.Subscript()
    var_13 = var_3.visit_Subscript(var_12)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_globals_skips_const_assignment_when_already_present. Retrieved 11/39 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONSTANT'
    var_2 = 'pkg.module'
    var_3 = 'MY_CONSTANT'
    var_4 = 10
    var_5 = module_1.Constant()
    var_6 = None
    var_7 = 20
    var_8 = module_1.Constant()
    var_9 = 'pkg.module'
    var_10 = module_1.Constant()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_globals_predicate_false_by_multiple_targets. Retrieved 6/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = module_1.Constant()
    var_5 = 'pkg'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_visit_Constant_valid_string_name_resolution. Retrieved 8/11 statements.
# Partially parsed test_visit_Constant_valid_string_no_alias. Retrieved 6/9 statements.
# Partially parsed test_visit_Constant_valid_string_expression. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = '['
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'pkg.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = module_1.Constant()
    var_7 = var_4.visit_Constant(var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'SimpleName'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = '1 + 2'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = var_5.op



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

def test_case_0():
    var_0 = True
    var_1 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_parser_compile_empty. Retrieved 4/5 statements.
# Partially parsed test_parser_compile_with_content. Retrieved 3/7 statements.
# Partially parsed test_parser_compile_with_toc_and_link. Retrieved 3/10 statements.
# Partially parsed test_parser_compile_filtering_private. Retrieved 4/10 statements.
# Partially parsed test_parser_compile_with_constants. Retrieved 4/10 statements.
# Partially parsed test_parser_compile_ignores_magic_methods_in_doc_list. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, toc=var_0, level=var_0)
    var_2 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, toc=var_0, level=var_0)
    var_2 = var_1.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = var_2.compile()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, toc=var_1, level=var_0)
    var_3 = var_2.compile()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_attr_predicate_false_when_attribute_exists. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'SubObject'
    var_1 = ()
    var_2 = 'b'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'a.b'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_globals_predicate_false. Retrieved 6/27 statements.


import ast as module_0

def test_case_0():
    var_0 = 'PKG_VAL'
    var_1 = 123
    var_2 = module_0.Constant()
    var_3 = 'mock'
    var_4 = '123'
    var_5 = 'pkg'



