####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_imports_method. Retrieved 9/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module_name'
    var_5 = 'alias_name'
    var_6 = 'function_name'
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_table_single_row. Retrieved 7/8 statements.
# Partially parsed test_table_multiple_rows. Retrieved 10/11 statements.
# Partially parsed test_table_single_item. Retrieved 6/7 statements.
# Partially parsed test_table_empty_items. Retrieved 4/5 statements.
# Partially parsed test_table_with_single_string_item. Retrieved 5/6 statements.
# Partially parsed test_table_with_varying_length_titles. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = '| a | b |\n|:---:|:---:|\n| c | d |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = [var_2, var_3]
    var_5 = 'e'
    var_6 = 'f'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = '| a | b |\n|:---:|:---:|\n| c | d |\n| e | f |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = '| a | b |\n|:---:|:---:|\n| c |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = '| a | b |\n|:---:|:---:|\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_2]
    var_4 = '| a | b |\n|:---:|:---:|\n| c |\n\n'

def test_case_0():
    var_0 = 'abc'
    var_1 = 'd'
    var_2 = 'e'
    var_3 = 'f'
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = '| abc | d |\n|:-----:|:---:|\n| e | f |\n\n'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_visit_Constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_invalid_syntax_string. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_valid_expression_string. Retrieved 4/9 statements.
# Partially parsed test_visit_Constant_with_self_ty_replacement. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'some_name'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'self_type'
    var_3 = module_0.Resolver(var_0, var_1, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 8/10 statements.
# Partially parsed test_globals_with_ann_assign_no_value. Retrieved 7/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = module_1.Name()
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'y'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 'hello'
    var_6 = module_1.Constant()
    var_7 = 'str'
    var_8 = module_1.Assign()
    var_9 = var_0.globals(var_1, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'z'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 3.14
    var_6 = module_1.Constant()
    var_7 = module_1.Assign()
    var_8 = var_0.globals(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 2
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Tuple()
    var_11 = module_1.Assign()
    var_12 = var_0.globals(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'b'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 'a'
    var_6 = module_1.Constant()
    var_7 = module_1.Constant()
    var_8 = [var_6, var_7]
    var_9 = module_1.List()
    var_10 = module_1.Assign()
    var_11 = var_0.globals(var_1, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'c'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = True
    var_6 = module_1.Constant()
    var_7 = False
    var_8 = module_1.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_1.Set()
    var_11 = module_1.Assign()
    var_12 = var_0.globals(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'd'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = 'value'
    var_9 = module_1.Constant()
    var_10 = [var_9]
    var_11 = module_1.Dict()
    var_12 = module_1.Assign()
    var_13 = var_0.globals(var_1, var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'e'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 'int'
    var_6 = module_1.Name()
    var_7 = '42'
    var_8 = module_1.Constant()
    var_9 = [var_8]
    var_10 = module_1.Call(*var_9)
    var_11 = module_1.Assign()
    var_12 = var_0.globals(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'obj'
    var_3 = module_1.Name()
    var_4 = 'attr'
    var_5 = module_1.Attribute()
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = module_1.Assign()
    var_10 = var_0.globals(var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = module_1.Name()
    var_4 = 'y'
    var_5 = module_1.Name()
    var_6 = [var_3, var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = module_1.Assign()
    var_10 = var_0.globals(var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = module_1.Name()
    var_4 = None
    var_5 = 'int'
    var_6 = module_1.Name()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 11/17 statements.
# Partially parsed test_compile_without_toc. Retrieved 11/17 statements.
# Partially parsed test_compile_with_constants. Retrieved 9/15 statements.
# Partially parsed test_compile_with_missing_documentation. Retrieved 10/16 statements.
# Partially parsed test_compile_with_immediate_family. Retrieved 13/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.func'
    var_4 = '# Module `module`\n\n'
    var_5 = '## func()\n\n'
    var_6 = 'Docstring for func'
    var_7 = 0
    var_8 = set()
    var_9 = var_1.compile()
    var_10 = '**Table of contents:**\n    + [`module`](#module)\n        + [`module.func`](#module-func)\n\n# Module `module`\n\n\n\n## func()\n\nDocstring for func\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.func'
    var_4 = '# Module `module`\n\n'
    var_5 = '## func()\n\n'
    var_6 = 'Docstring for func'
    var_7 = 1
    var_8 = set()
    var_9 = var_1.compile()
    var_10 = '# Module `module`\n\n\n\n## func()\n\nDocstring for func\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n\n'
    var_4 = 'module.CONST'
    var_5 = 'int'
    var_6 = set()
    var_7 = var_1.compile()
    var_8 = '# Module `module`\n\n\n| Constants | Type |\n| --- | --- |\n| `CONST` | `int` |\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.__magic'
    var_4 = '# Module `module`\n\n'
    var_5 = '## __magic()\n\n'
    var_6 = 1
    var_7 = set()
    var_8 = var_1.compile()
    var_9 = '# Module `module`\n\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.func'
    var_4 = 'module.alias'
    var_5 = '# Module `module`\n\n'
    var_6 = '## func()\n\n'
    var_7 = '## alias()\n\n'
    var_8 = 'Docstring for func'
    var_9 = 1
    var_10 = set()
    var_11 = var_1.compile()
    var_12 = '# Module `module`\n\n\n\n## func()\n\nDocstring for func\n'



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = set()

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'root'
    var_3 = 'import os'
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'root'
    var_3 = 'import os'
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Parser(b_level=var_0)
    var_2 = 'root'
    var_3 = 'import os'
    var_4 = var_1.parse(var_2, var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'from os import path'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'x = 10'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'x: int = 10'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'def func(): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class MyClass: pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '"""This is a docstring"""'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #8
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = module_0.Parser()
    var_5 = 'class A(B, C): pass'
    var_6 = var_4.parse(var_1, var_5)
    var_7 = module_0.Parser()
    var_8 = 'class A:\n    x: int\n    y: str'
    var_9 = var_7.parse(var_1, var_8)
    var_10 = module_0.Parser()
    var_11 = 'class A:\n    class B: pass'
    var_12 = var_10.parse(var_1, var_11)
    var_13 = module_0.Parser()
    var_14 = 'class A:\n    def f(): pass'
    var_15 = var_13.parse(var_1, var_14)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_walk_body_with_try_stmt. Retrieved 8/14 statements.
# Partially parsed test_walk_body_with_nested_stmts. Retrieved 12/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt'
    var_1 = [var_0]
    var_2 = module_0.walk_body(var_1)
    var_3 = list(var_2)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'body_stmt'
    var_1 = [var_0]
    var_2 = 'orelse_stmt'
    var_3 = [var_2]
    var_4 = module_0.If()
    var_5 = [var_4]
    var_6 = module_1.walk_body(var_5)
    var_7 = list(var_6)

def test_case_0():
    var_0 = 'body_stmt'
    var_1 = [var_0]
    var_2 = 'handler_body'
    var_3 = [var_2]
    var_4 = 'orelse_stmt'
    var_5 = [var_4]
    var_6 = 'finalbody_stmt'
    var_7 = [var_6]

import ast as module_0

def test_case_0():
    var_0 = 'nested_body'
    var_1 = [var_0]
    var_2 = 'nested_orelse'
    var_3 = [var_2]
    var_4 = module_0.If()
    var_5 = [var_4]
    var_6 = 'handler_body'
    var_7 = [var_6]
    var_8 = 'orelse_stmt'
    var_9 = [var_8]
    var_10 = 'finalbody_stmt'
    var_11 = [var_10]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'stmt2'
    var_2 = [var_0, var_1]
    var_3 = module_0.walk_body(var_2)
    var_4 = list(var_3)



# Parsed testcases at query #10
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = module_1.Name()
    var_3 = None
    var_4 = module_1.AnnAssign()
    var_5 = 'root'
    var_6 = 'name'
    var_7 = []
    var_8 = [var_4]
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)



# Parsed testcases at query #11
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 'y'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = [var_3, var_6]
    var_8 = 42
    var_9 = module_1.Constant()
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_23_evaluates_to_false. Retrieved 6/11 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = module_0.Constant()
    var_3 = 'int'
    var_4 = module_1.Parser()
    var_5 = 'test'



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ENUM_VALUE_1'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = None
    var_6 = module_1.AnnAssign()
    var_7 = 'ENUM_VALUE_2'
    var_8 = module_1.Name()
    var_9 = module_1.Name()
    var_10 = module_1.AnnAssign()
    var_11 = 'ENUM_VALUE_3'
    var_12 = module_1.Name()
    var_13 = module_1.Name()
    var_14 = module_1.AnnAssign()
    var_15 = [var_6, var_10, var_14]
    var_16 = 'root'
    var_17 = 'name'
    var_18 = 'enum.Enum'
    var_19 = module_1.Name()
    var_20 = [var_19]
    var_21 = var_0.class_api(var_16, var_17, var_20, var_15)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_predicate_evaluates_to_false. Retrieved 8/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module_name'
    var_1 = 'ClassName'
    var_2 = []
    var_3 = []
    var_4 = module_0.Parser()
    var_5 = var_4.class_api(var_0, var_1, var_2, var_3)
    var_6 = 0
    var_7 = var_3[var_6]



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__magic__.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_name_in_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_root_name. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.name'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root._name'
    var_4 = var_0.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.__name__'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.name'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = {var_1}
    var_3 = 'root.name'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = var_0.is_public(var_1)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_1.arguments(*var_5)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_9, var_2, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = 'y'
    var_6 = module_1.arg()
    var_7 = [var_4, var_6]
    var_8 = []
    var_9 = []
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = [var_3, var_11]
    var_13 = module_1.arguments(*var_7)
    var_14 = 'root'
    var_15 = 'name'
    var_16 = False
    var_17 = var_0.func_api(var_14, var_15, var_13, var_3, has_self=var_16, cls_method=var_16)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'args'
    var_4 = None
    var_5 = module_1.arg()
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_1.arguments(*var_2)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_9, var_4, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'x'
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = [var_3]
    var_8 = []
    var_9 = module_1.arguments(*var_2)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_9, var_3, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = module_1.arg()
    var_8 = []
    var_9 = module_1.arguments(*var_2)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_9, var_3, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = module_1.arguments(*var_2)
    var_8 = 'root'
    var_9 = 'name'
    var_10 = 'int'
    var_11 = module_1.Load()
    var_12 = module_1.Name()
    var_13 = False
    var_14 = var_0.func_api(var_8, var_9, var_7, var_12, has_self=var_13, cls_method=var_13)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_1.arguments(*var_5)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = True
    var_13 = var_0.func_api(var_10, var_11, var_9, var_3, has_self=var_12, cls_method=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = module_1.arguments(*var_5)
    var_10 = 'root'
    var_11 = 'name'
    var_12 = True
    var_13 = False
    var_14 = var_0.func_api(var_10, var_11, var_9, var_3, has_self=var_12, cls_method=var_13)



# Parsed testcases at query #19
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'def example_function(): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'async def example_async_function(): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass: pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '@decorator\ndef example_function(): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass(BaseClass): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass:\n    member: int = 1'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass(enum.Enum):\n    ENUM = 1'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass:\n    def method(self): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass:\n    @classmethod\n    def method(cls): pass'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class ExampleClass:\n    @staticmethod\n    def method(): pass'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_public_returns_true_when_s_is_root_and_all_l_contains_s. Retrieved 4/6 statements.
# Partially parsed test_is_public_returns_true_when_parent_s_in_all_l. Retrieved 5/7 statements.
# Partially parsed test_is_public_returns_true_when_s_is_public_family_and_all_l_is_empty. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.submodule'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = set()
    var_3 = 'module.public_function'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = module_1.AnnAssign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 100
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'root'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'test'
    var_5 = module_1.Constant()
    var_6 = 'str'
    var_7 = module_1.Assign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = 2
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Tuple()
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'b'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 3
    var_5 = module_1.Constant()
    var_6 = 4
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.List()
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'c'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 5
    var_5 = module_1.Constant()
    var_6 = 6
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Set()
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'd'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'key'
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = 'value'
    var_8 = module_1.Constant()
    var_9 = [var_8]
    var_10 = module_1.Dict()
    var_11 = module_1.Assign()
    var_12 = 'root'
    var_13 = var_0.globals(var_12, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'e'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'str'
    var_5 = module_1.Name()
    var_6 = 'test'
    var_7 = module_1.Constant()
    var_8 = [var_7]
    var_9 = module_1.Call(*var_8)
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'f'
    var_2 = module_1.Name()
    var_3 = 'g'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 123
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = module_1.Name()
    var_3 = 'attr'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 456
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'h'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'x'
    var_5 = module_1.Name()
    var_6 = module_1.Assign()
    var_7 = 'root'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 789
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'root'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 999
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'root'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'public_func'
    var_5 = module_1.Constant()
    var_6 = 'PublicClass'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.Tuple()
    var_10 = module_1.Assign()
    var_11 = 'root'
    var_12 = var_0.globals(var_11, var_10)



# Parsed testcases at query #22
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test|value'
    var_1 = 'test&value'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__e_type_single_empty_element. Retrieved 1/2 statements.
# Failed to parse test__e_type_single_element_with_non_constant.
# Partially parsed test__e_type_single_element_with_constants_of_same_type. Retrieved 2/9 statements.
# Partially parsed test__e_type_single_element_with_constants_of_different_types. Retrieved 2/9 statements.
# Partially parsed test__e_type_multiple_elements_with_constants. Retrieved 4/14 statements.
# Partially parsed test__e_type_multiple_elements_with_mixed_constants. Retrieved 4/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_imports_predicate_false. Retrieved 4/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'alias_name'
    var_3 = 'root'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 7/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.key'
    var_2 = 'existing_value'
    var_3 = 'KEY'
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = 'root'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 8/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 6/11 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 9/14 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 6/13 statements.
# Partially parsed test_globals_with_uppercase_name. Retrieved 5/10 statements.
# Partially parsed test_globals_with___all__. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 2
    var_3 = module_1.Constant()
    var_4 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3
    var_3 = module_1.Constant()
    var_4 = 'float'
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 0
    var_5 = module_1.Constant()
    var_6 = 4
    var_7 = module_1.Constant()
    var_8 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 5
    var_4 = module_1.Constant()
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 6
    var_3 = module_1.Constant()
    var_4 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'x'
    var_3 = module_1.Constant()
    var_4 = 'y'
    var_5 = module_1.Constant()
    var_6 = [var_3, var_5]
    var_7 = module_1.List()
    var_8 = 'root'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_visit_Constant_with_valid_string. Retrieved 9/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'valid_string'
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)
    var_6 = 0
    var_7 = module_1.parse(var_3)
    var_8 = var_7.body[var_6]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_is_enum_evaluates_to_true_when_bases_contain_enum. Retrieved 9/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'test_module'
    var_3 = 'enum.Enum'
    var_4 = [var_3]
    var_5 = []
    var_6 = var_0.class_api(var_2, var_1, var_4, var_5)
    var_7 = var_0.doc[var_1]
    var_8 = 'enum.'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_class_api. Retrieved 13/40 statements.


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
    var_11 = '_private_attr'
    var_12 = 'attr3'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 15/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = var_8.targets
    var_10 = len(var_9)
    var_11 = 1
    var_12 = var_10 == var_11
    var_13 = 0
    var_14 = var_8.targets[var_13]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 5/10 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 8/13 statements.
# Partially parsed test_visit_Name_with_typevar_alias. Retrieved 8/13 statements.
# Partially parsed test_visit_Name_with_no_alias. Retrieved 6/11 statements.
# Partially parsed test_visit_Name_with_self_reference_alias. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'SelfType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.Alias'
    var_2 = 'AliasType'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'Alias'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.TypeVar'
    var_2 = 'typing.TypeVar'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'TypeVar'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'NoAlias'
    var_5 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.Alias'
    var_2 = {var_1: var_1}
    var_3 = ''
    var_4 = module_0.Resolver(var_0, var_2, var_3)
    var_5 = 'Alias'
    var_6 = module_1.Load()



# Parsed testcases at query #32
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'a'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'a'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'b'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = 'b'
    var_7 = None
    var_8 = module_1.arg()
    var_9 = [var_5, var_8]
    var_10 = 'root'
    var_11 = False
    var_12 = var_0.func_ann(var_10, var_9, has_self=var_11, cls_method=var_11)
    var_13 = list(var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'a'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_e_type_empty_sequence. Retrieved 1/2 statements.
# Partially parsed test_e_type_none_element. Retrieved 2/3 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 2/3 statements.
# Partially parsed test_e_type_mixed_types. Retrieved 2/6 statements.
# Partially parsed test_e_type_single_constant. Retrieved 1/4 statements.
# Partially parsed test_e_type_multiple_constants_same_type. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_class_api_with_enum. Retrieved 10/16 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/13 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 7/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.MyClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = []
    var_8 = var_0.class_api(var_1, var_2, var_6, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.MyClass'
    var_3 = 'enum.Enum'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 'ENUM_VALUE'
    var_8 = 1
    var_9 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.MyClass'
    var_3 = []
    var_4 = 'member'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.MyClass'
    var_3 = []
    var_4 = 'member'
    var_5 = 1
    var_6 = module_1.Constant()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_public_returns_true_when_s_is_root_and_all_l_contains_s. Retrieved 4/8 statements.
# Partially parsed test_is_public_returns_true_when_parent_s_is_in_all_l. Retrieved 6/10 statements.
# Partially parsed test_is_public_returns_true_when_s_is_public_family_and_all_l_is_empty. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = {var_1}
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'root.child'
    var_5 = var_0.is_public(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.public_method'
    var_4 = var_0.is_public(var_3)
    assert var_4 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 12/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 10
    var_6 = module_1.Constant()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 20
    var_10 = module_1.Constant()
    var_11 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_globals_assign_non_constant_value. Retrieved 9/13 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = 'int'
    var_3 = module_0.Name()
    var_4 = 10
    var_5 = module_0.Constant()
    var_6 = module_0.AnnAssign()
    var_7 = module_1.Parser()
    var_8 = 'root'
    var_9 = var_7.globals(var_8, var_6)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 10
    var_4 = module_0.Constant()
    var_5 = module_0.Assign()
    var_6 = module_1.Parser()
    var_7 = 'root'
    var_8 = var_6.globals(var_7, var_5)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 10
    var_4 = module_0.Constant()
    var_5 = 'int'
    var_6 = module_0.Assign()
    var_7 = module_1.Parser()
    var_8 = 'root'
    var_9 = var_7.globals(var_8, var_6)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = 'y'
    var_3 = module_0.Name()
    var_4 = [var_1, var_3]
    var_5 = 10
    var_6 = module_0.Constant()
    var_7 = module_0.Assign()
    var_8 = module_1.Parser()
    var_9 = 'root'
    var_10 = var_8.globals(var_9, var_7)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'obj'
    var_1 = module_0.Name()
    var_2 = 'x'
    var_3 = module_0.Attribute()
    var_4 = [var_3]
    var_5 = 10
    var_6 = module_0.Constant()
    var_7 = module_0.Assign()
    var_8 = module_1.Parser()
    var_9 = 'root'
    var_10 = var_8.globals(var_9, var_7)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 1
    var_4 = module_0.Constant()
    var_5 = 2
    var_6 = module_0.Constant()
    var_7 = module_1.Parser()
    var_8 = 'root'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 'x'
    var_4 = module_0.Constant()
    var_5 = 'y'
    var_6 = module_0.Constant()
    var_7 = [var_4, var_6]
    var_8 = module_0.List()
    var_9 = module_0.Assign()
    var_10 = module_1.Parser()
    var_11 = 'root'
    var_12 = var_10.globals(var_11, var_9)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_visit_name_returns_self_when_node_id_matches_self_ty. Retrieved 8/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'self_ty'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)
    var_7 = var_6.ctx



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_visit_Subscript_with_Union. Retrieved 23/24 statements.
# Partially parsed test_visit_Subscript_with_Optional. Retrieved 16/18 statements.
# Partially parsed test_visit_Subscript_with_PEP585. Retrieved 18/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing.Union'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'A'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'B'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_8, var_11]
    var_13 = module_1.Load()
    var_14 = module_1.Tuple()
    var_15 = module_1.Load()
    var_16 = module_1.Subscript()
    var_17 = var_2.visit_Subscript(var_16)
    var_18 = module_1.Load()
    var_19 = module_1.Name()
    var_20 = module_1.BitOr()
    var_21 = module_1.Load()
    var_22 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing.Optional'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'A'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)
    var_12 = module_1.Load()
    var_13 = module_1.Name()
    var_14 = module_1.BitOr()
    var_15 = None

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing.List'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'A'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = module_1.Load()
    var_10 = module_1.Subscript()
    var_11 = var_2.visit_Subscript(var_10)
    var_12 = 'list'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = module_1.Load()
    var_16 = module_1.Name()
    var_17 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing.Dict'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'A'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'B'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = [var_8, var_11]
    var_13 = module_1.Load()
    var_14 = module_1.Tuple()
    var_15 = module_1.Load()
    var_16 = module_1.Subscript()
    var_17 = var_2.visit_Subscript(var_16)



# Parsed testcases at query #40
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os as operating_system'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path as p'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'from .. import utils'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.sub'
    var_2 = 'from .. import utils'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os, sys as system'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path, sep as separator'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_class_api_is_public_family_false. Retrieved 7/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = ''



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_evaluates_to_true_when_types_differ. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #43
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.function'
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = 'kwarg'
    var_6 = module_0.arg()
    var_7 = [var_6]
    var_8 = [var_4]
    var_9 = []
    var_10 = module_0.arguments(*var_3)
    var_11 = None
    var_12 = False
    var_13 = False
    var_14 = module_1.Parser()
    var_15 = var_14.func_api(var_0, var_1, var_10, var_11, has_self=var_12, cls_method=var_13)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_func_api_has_default_is_false. Retrieved 18/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = [var_9]
    var_11 = module_1.arguments(*var_5)
    var_12 = 'root'
    var_13 = 'name'
    var_14 = False
    var_15 = var_0.func_api(var_12, var_13, var_11, var_3, has_self=var_14, cls_method=var_14)
    var_16 = module_1.Constant()
    var_17 = [var_3, var_16, var_3]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_visit_name_with_alias_and_not_in_alias_of_itself. Retrieved 7/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_is_public_family_evaluates_to_false_in_class_api. Retrieved 9/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private_attr'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = '__magic_attr__'
    var_5 = module_1.Constant()
    var_6 = 'root'
    var_7 = 'name'
    var_8 = []



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_is_magic_predicate_evaluates_to_true. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = 'docstring'
    var_3 = 0
    var_4 = set()
    var_5 = var_0.compile()
    assert var_5 == '\n'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 6/11 statements.
# Partially parsed test_globals_with_assign_no_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_non_uppercase_var. Retrieved 6/11 statements.
# Partially parsed test_globals_with___all__. Retrieved 8/13 statements.
# Partially parsed test_globals_with_invalid___all__. Retrieved 5/10 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 6/13 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'int'
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'int'
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'VAR'
    var_3 = module_1.Constant()
    var_4 = [var_3]
    var_5 = module_1.Load()
    var_6 = module_1.List()
    var_7 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR1'
    var_2 = 'VAR2'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'list'
    var_2 = module_1.Load()
    var_3 = module_1.Name()
    var_4 = 0
    var_5 = module_1.Constant()
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = 'root'



# Parsed testcases at query #2
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 42
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
    var_0 = []
    var_1 = module_0.Load()
    var_2 = module_0.Tuple()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Load()
    var_6 = module_0.Tuple()
    var_7 = module_1.const_type(var_6)
    assert var_7 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Load()
    var_2 = module_0.List()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'list[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Load()
    var_6 = module_0.List()
    var_7 = module_1.const_type(var_6)
    assert var_7 == 'list[str, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Set()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'set[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1.0
    var_1 = module_0.Constant()
    var_2 = 2.0
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Set()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'set[float, float]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0.Dict()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'dict[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = 'a'
    var_6 = module_0.Constant()
    var_7 = 'b'
    var_8 = module_0.Constant()
    var_9 = [var_6, var_8]
    var_10 = module_0.Dict()
    var_11 = module_1.const_type(var_10)
    assert var_11 == 'dict[int, int, str, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = 42
    var_4 = module_0.Constant()
    var_5 = [var_4]
    var_6 = []
    var_7 = module_0.Call(*var_5)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = 'hello'
    var_4 = module_0.Constant()
    var_5 = [var_4]
    var_6 = []
    var_7 = module_0.Call(*var_5)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown'
    var_1 = module_0.Load()
    var_2 = module_0.Name()
    var_3 = 42
    var_4 = module_0.Constant()
    var_5 = [var_4]
    var_6 = []
    var_7 = module_0.Call(*var_5)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'Any'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 12/17 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 12/18 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 12/21 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 12/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'B'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = [var_7]
    var_9 = []
    var_10 = var_0.class_api(var_1, var_4, var_8, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'enum.Enum'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = [var_7]
    var_9 = []
    var_10 = var_0.class_api(var_1, var_4, var_8, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = None
    var_9 = 1
    var_10 = 'test_module.A'
    var_11 = []

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'X'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = 'test_module.A'
    var_8 = 'enum.Enum'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = [var_10]

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = None
    var_9 = 1
    var_10 = 'test_module.A'
    var_11 = []

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = None
    var_9 = 1
    var_10 = 'test_module.A'
    var_11 = []



# Parsed testcases at query #4
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = set()
    var_5 = {var_1: var_4}

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1\ny = 2'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = set()
    var_5 = {var_2: var_4}

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "'''Some docstring'''\nx = 1"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = set()
    var_5 = {var_2: var_4}

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nx = 1'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = set()
    var_5 = {var_2: var_4}

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = set()
    var_5 = {var_2: var_4}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_annassign_with_name_target_and_non_none_value. Retrieved 9/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Name()
    var_7 = module_1.AnnAssign()
    var_8 = var_7.target



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os as operating_system'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path as p'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'from ..sub import func'
    var_3 = var_0.parse(var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os, sys as system'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_globals_predicate_evaluates_to_false. Retrieved 6/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'root'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_visit_Subscript_with_Union. Retrieved 11/21 statements.
# Partially parsed test_visit_Subscript_with_Optional. Retrieved 8/17 statements.
# Partially parsed test_visit_Subscript_with_PEP585_deprecated. Retrieved 8/15 statements.
# Partially parsed test_visit_Subscript_with_unknown_type. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = module_1.Load()
    var_5 = 'A'
    var_6 = module_1.Load()
    var_7 = 'B'
    var_8 = module_1.Load()
    var_9 = module_1.Load()
    var_10 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Optional'
    var_4 = module_1.Load()
    var_5 = 'A'
    var_6 = module_1.Load()
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = 'A'
    var_6 = module_1.Load()
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Unknown'
    var_4 = module_1.Load()
    var_5 = 'A'
    var_6 = module_1.Load()
    var_7 = module_1.Load()



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'def func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = module_1.FunctionDef(*var_5)
    var_10 = var_0.api(var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'async def func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = module_1.AsyncFunctionDef(*var_5)
    var_10 = var_0.api(var_1, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class Cls: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'Cls'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.ClassDef()
    var_9 = var_0.api(var_1, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '@decorator\ndef func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = 'decorator'
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = None
    var_11 = module_1.FunctionDef(*var_5)
    var_12 = var_0.api(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '@classmethod\ndef func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = 'classmethod'
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = None
    var_11 = module_1.FunctionDef(*var_5)
    var_12 = 'Cls'
    var_13 = var_0.api(var_1, var_11, prefix=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '@staticmethod\ndef func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = 'staticmethod'
    var_8 = module_1.Name()
    var_9 = [var_8]
    var_10 = None
    var_11 = module_1.FunctionDef(*var_5)
    var_12 = 'Cls'
    var_13 = var_0.api(var_1, var_11, prefix=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'def func():\n    """docstring"""'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'
    var_5 = module_1.arguments()
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = module_1.FunctionDef(*var_5)
    var_10 = var_0.api(var_1, var_9)



# Parsed testcases at query #10
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.func'
    var_3 = 'posonly'
    var_4 = module_1.arg()
    var_5 = [var_4]
    var_6 = 'arg1'
    var_7 = module_1.arg()
    var_8 = 'arg2'
    var_9 = module_1.arg()
    var_10 = [var_7, var_9]
    var_11 = 'vararg'
    var_12 = module_1.arg()
    var_13 = 'kwonly1'
    var_14 = module_1.arg()
    var_15 = 'kwonly2'
    var_16 = module_1.arg()
    var_17 = [var_14, var_16]
    var_18 = None
    var_19 = [var_18, var_18]
    var_20 = 'kwarg'
    var_21 = module_1.arg()
    var_22 = [var_18, var_18]
    var_23 = module_1.arguments(*var_10)
    var_24 = None
    var_25 = True
    var_26 = var_0.func_api(var_1, var_2, var_23, var_24, has_self=var_25, cls_method=var_25)
    var_27 = '#### func()\n\n*Full name:* `module.func`\n\n| posonly | / | arg1 | arg2 | *vararg | * | kwonly1 | kwonly2 | **kwarg | return |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n| Self |  | Self | Self | Self |  | Self | Self | Self |  |\n'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_globals_const_condition_false. Retrieved 6/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'TEST'
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'int'



# Parsed testcases at query #12
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_name'
    var_3 = []
    var_4 = 'test_attr'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = module_1.AnnAssign()
    var_9 = [var_8]
    var_10 = var_0.class_api(var_1, var_2, var_3, var_9)



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = module_1.FunctionDef()
    var_6 = 'root'
    var_7 = var_1.api(var_6, var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_visit_Subscript_handles_typing_Union. Retrieved 11/25 statements.
# Partially parsed test_visit_Subscript_handles_typing_Optional. Retrieved 8/19 statements.
# Partially parsed test_visit_Subscript_handles_PEP585_deprecated_names. Retrieved 8/17 statements.
# Partially parsed test_visit_Subscript_returns_node_for_non_typing_Union_or_Optional. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = module_1.Load()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Load()
    var_10 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Optional'
    var_4 = module_1.Load()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'List'
    var_4 = module_1.Load()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Dict'
    var_4 = module_1.Load()
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Load()



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'posonlyarg'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'arg1'
    var_6 = module_1.arg()
    var_7 = [var_6]
    var_8 = 'kwonlyarg'
    var_9 = module_1.arg()
    var_10 = [var_9]
    var_11 = [var_2]
    var_12 = 'default_value'
    var_13 = module_1.Load()
    var_14 = module_1.Name()
    var_15 = [var_14]
    var_16 = module_1.arguments(*var_7)
    var_17 = 'root'
    var_18 = 'name'
    var_19 = False
    var_20 = var_0.func_api(var_17, var_18, var_16, var_2, has_self=var_19, cls_method=var_19)



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a & b'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a | b'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_visit_Attribute_with_typing_prefix. Retrieved 11/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
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

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'module'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'SomeClass'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_public_with_public_family_name. Retrieved 3/4 statements.
# Partially parsed test_is_public_with_private_family_name. Retrieved 3/4 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/4 statements.
# Partially parsed test_is_public_with_name_in_all_list. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_name_not_in_all_list. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_child_of_root_in_all_list. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_parent_in_all_list. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.public'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test._private'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.__magic__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.public'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.other'
    var_2 = 'test.module.public'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.child'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.child'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = module_1.Constant()
    var_5 = 'x'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = module_1.Load()
    var_9 = module_1.Subscript()
    var_10 = var_2.visit_Subscript(var_9)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 9/15 statements.
# Partially parsed test_compile_without_toc. Retrieved 10/16 statements.
# Partially parsed test_compile_with_constants. Retrieved 12/18 statements.
# Partially parsed test_compile_with_missing_docstring. Retrieved 9/15 statements.
# Partially parsed test_compile_with_magic_method. Retrieved 10/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module.name'
    var_3 = '## Module `module.name`\n\n'
    var_4 = 'Docstring content'
    var_5 = 'module'
    var_6 = set()
    var_7 = var_1.compile()
    var_8 = '**Table of contents:**\n    + [module.name](#module-name)\n\n## Module `module.name`\n\nDocstring content\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module.name'
    var_3 = '## Module `module.name`\n\n'
    var_4 = 'Docstring content'
    var_5 = 1
    var_6 = 'module'
    var_7 = set()
    var_8 = var_1.compile()
    var_9 = '## Module `module.name`\n\nDocstring content\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module.name'
    var_3 = '## Module `module.name`\n\n'
    var_4 = 'Docstring content'
    var_5 = 1
    var_6 = 'module'
    var_7 = 'module.name.constant'
    var_8 = 'str'
    var_9 = {var_2}
    var_10 = var_1.compile()
    var_11 = '## Module `module.name`\n\n| Constants | Type |\n|-----------|------|\n| constant | str |\n\nDocstring content\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module.name'
    var_3 = '## Module `module.name`\n\n'
    var_4 = 1
    var_5 = 'module'
    var_6 = set()
    var_7 = var_1.compile()
    var_8 = '## Module `module.name`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module.__magic__'
    var_3 = '## Module `module.__magic__`\n\n'
    var_4 = 'Docstring content'
    var_5 = 1
    var_6 = 'module'
    var_7 = set()
    var_8 = var_1.compile()
    var_9 = ''



# Parsed testcases at query #23
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)

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
    var_7 = 'root'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = False
    var_7 = var_0.func_ann(var_5, var_4, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = False
    var_7 = var_0.func_ann(var_5, var_4, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = 'x'
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = module_1.arg()
    var_11 = 'y'
    var_12 = None
    var_13 = module_1.arg()
    var_14 = '*'
    var_15 = module_1.arg()
    var_16 = 'z'
    var_17 = 'str'
    var_18 = module_1.Load()
    var_19 = module_1.Name()
    var_20 = module_1.arg()
    var_21 = [var_5, var_10, var_13, var_15, var_20]
    var_22 = 'root'
    var_23 = True
    var_24 = False
    var_25 = var_0.func_ann(var_22, var_21, has_self=var_23, cls_method=var_24)
    var_26 = list(var_25)



# Parsed testcases at query #24
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
    var_6 = module_1.FunctionDef(*var_3)
    var_7 = var_0.api(var_1, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = module_1.arguments()
    var_4 = []
    var_5 = []
    var_6 = module_1.AsyncFunctionDef(*var_3)
    var_7 = var_0.api(var_1, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = module_1.ClassDef()
    var_7 = var_0.api(var_1, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'decorator'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'test_func'
    var_6 = module_1.arguments()
    var_7 = []
    var_8 = [var_4]
    var_9 = module_1.FunctionDef(*var_6)
    var_10 = var_0.api(var_1, var_9)

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
    var_8 = 'TestClass'
    var_9 = var_0.api(var_1, var_7, prefix=var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'test_module'
    var_3 = 'test_func'
    var_4 = module_1.arguments()
    var_5 = []
    var_6 = []
    var_7 = module_1.FunctionDef(*var_4)
    var_8 = var_1.api(var_2, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'This is a test function'
    var_3 = 'test_func'
    var_4 = module_1.arguments()
    var_5 = module_1.Constant()
    var_6 = module_1.Expr()
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.FunctionDef(*var_4)
    var_10 = var_0.api(var_1, var_9)



# Parsed testcases at query #25
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = 'int'
    var_6 = module_1.Name()
    var_7 = module_1.AnnAssign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)
    var_10 = var_0.alias
    var_11 = len(var_10)
    assert var_11 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)
    var_11 = var_0.alias
    var_12 = len(var_11)
    assert var_12 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = module_1.Name()
    var_3 = 'attr'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)
    var_11 = var_0.alias
    var_12 = len(var_11)
    assert var_12 == 0



# Parsed testcases at query #26
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'x'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'y'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Name()
    var_4 = module_1.arg()
    var_5 = 'y'
    var_6 = 'str'
    var_7 = module_1.Name()
    var_8 = module_1.arg()
    var_9 = [var_4, var_8]
    var_10 = 'root'
    var_11 = False
    var_12 = var_0.func_ann(var_10, var_9, has_self=var_11, cls_method=var_11)
    var_13 = list(var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'x'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = False
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'MyClass'
    var_3 = module_1.Name()
    var_4 = module_1.arg()
    var_5 = 'x'
    var_6 = None
    var_7 = module_1.arg()
    var_8 = [var_4, var_7]
    var_9 = 'root'
    var_10 = True
    var_11 = False
    var_12 = var_0.func_ann(var_9, var_8, has_self=var_10, cls_method=var_11)
    var_13 = list(var_12)



# Parsed testcases at query #27
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = None
    var_3 = 'kwarg'
    var_4 = module_0.arg()
    var_5 = [var_4]
    var_6 = [var_2]
    var_7 = []
    var_8 = module_0.arguments(*var_1)
    var_9 = module_1.Parser()
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_9.func_api(var_10, var_11, var_8, var_2, has_self=var_12, cls_method=var_12)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_e_type_single_element_empty_sequence. Retrieved 1/2 statements.
# Failed to parse test_e_type_single_element_non_constant.
# Partially parsed test_e_type_single_element_single_constant. Retrieved 3/6 statements.
# Partially parsed test_e_type_single_element_multiple_constants_same_type. Retrieved 4/7 statements.
# Failed to parse test_e_type_single_element_multiple_constants_different_types.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 5/8 statements.
# Failed to parse test_e_type_multiple_elements_different_types.
# Failed to parse test_e_type_multiple_elements_mixed_types.
# Partially parsed test_e_type_multiple_elements_with_empty_sequence. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

def test_case_0():
    var_0 = []

import ast as module_0

def test_case_0():
    var_0 = module_0.Constant()
    var_1 = [var_0]
    var_2 = [var_1]

import ast as module_0

def test_case_0():
    var_0 = module_0.Constant()
    var_1 = module_0.Constant()
    var_2 = [var_0, var_1]
    var_3 = [var_2]

import ast as module_0

def test_case_0():
    var_0 = module_0.Constant()
    var_1 = [var_0]
    var_2 = module_0.Constant()
    var_3 = [var_2]
    var_4 = [var_1, var_3]

import ast as module_0

def test_case_0():
    var_0 = module_0.Constant()
    var_1 = [var_0]
    var_2 = []
    var_3 = [var_1, var_2]



# Parsed testcases at query #29
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.module.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.module.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.module.__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.module.__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__magic__.__another_magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private._another_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__magic__._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_globals_const_get_returns_not_any. Retrieved 5/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'root'



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = [var_0]
    var_2 = module_0._e_type(*var_1)
    assert var_2 == ''



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_class_api. Retrieved 29/38 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.Class'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Base1'
    var_7 = 'Base2'
    var_8 = [var_6, var_7]
    var_9 = []
    var_10 = var_0.class_api(var_1, var_2, var_8, var_9)
    var_11 = 'attr1'
    var_12 = module_1.Name()
    var_13 = 'int'
    var_14 = module_1.Name()
    var_15 = None
    var_16 = module_1.AnnAssign()
    var_17 = [var_16]
    var_18 = []
    var_19 = var_0.class_api(var_1, var_2, var_18, var_17)
    var_20 = module_1.Name()
    var_21 = [var_20]
    var_22 = 1
    var_23 = module_1.Constant()
    var_24 = module_1.Assign()
    var_25 = [var_24]
    var_26 = 'enum.Enum'
    var_27 = [var_26]
    var_28 = var_0.class_api(var_1, var_2, var_27, var_25)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 5/10 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 8/13 statements.
# Partially parsed test_visit_Name_with_typevar. Retrieved 8/13 statements.
# Partially parsed test_visit_Name_with_no_alias. Retrieved 6/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'self_ty'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias_value'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.TypeVar'
    var_2 = 'typing.TypeVar'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'TypeVar'
    var_7 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'name'
    var_5 = module_1.Load()



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'enum.Enum'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.ClassDef()
    var_9 = 'test_root'
    var_10 = 'test_name'
    var_11 = module_1.Name()
    var_12 = [var_11]
    var_13 = []
    var_14 = var_0.class_api(var_9, var_10, var_12, var_13)



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_isinstance_node_Delete. Retrieved 2/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'



# Parsed testcases at query #37
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'other'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = 'other'
    var_5 = module_1.arg()
    var_6 = [var_3, var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_8)
    var_10 = list(var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = False
    var_7 = var_0.func_ann(var_5, var_4, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'param'
    var_2 = 'str'
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = False
    var_7 = var_0.func_ann(var_5, var_4, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)
    var_9 = var_0.resolve(var_5, var_2)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'param'
    var_2 = None
    var_3 = module_1.arg()
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = False
    var_7 = var_0.func_ann(var_5, var_4, has_self=var_6, cls_method=var_6)
    var_8 = list(var_7)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_public_returns_true_for_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_true_for_public_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_true_for_name_in_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_true_for_parent_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_returns_false_for_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_false_for_name_not_in_all. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root._name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'other'
    var_2 = 'root.name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_returns. Retrieved 14/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'x'
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = '*args'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'x'
    var_5 = 'y'
    var_6 = [var_2, var_2]
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = '**kwargs'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'a'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = 'a'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_globals_predicate_at_line_23_evaluates_to_false. Retrieved 7/17 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 'value'
    var_4 = 'type_comment'
    var_5 = module_1.Parser()
    var_6 = 'root'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_walk_body_with_if.
# Failed to parse test_walk_body_with_try.
# Failed to parse test_walk_body_without_control_structures.




# Parsed testcases at query #42
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_is_magic_predicate_evaluates_to_true. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = 'docstring'
    var_3 = 'root'
    var_4 = 1
    var_5 = set()
    var_6 = var_0.compile()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 10/12 statements.
# Partially parsed test_func_api_without_self_and_cls_method. Retrieved 10/12 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 12/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arg1'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = [var_2]
    var_6 = []
    var_7 = 'root'
    var_8 = 'name'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = '*args'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = '**kwargs'
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False



# Parsed testcases at query #45
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'y'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = module_1.Name()
    var_3 = 'attr'
    var_4 = module_1.Attribute()
    var_5 = [var_4]
    var_6 = 42
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = module_1.Name()
    var_3 = 'attr'
    var_4 = module_1.Attribute()
    var_5 = 'int'
    var_6 = module_1.Name()
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = module_1.AnnAssign()
    var_10 = 'root'
    var_11 = var_0.globals(var_10, var_9)



