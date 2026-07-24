####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test__m_with_empty_args.
# Partially parsed test__m_with_single_arg. Retrieved 1/2 statements.
# Partially parsed test__m_with_multiple_args. Retrieved 3/4 statements.
# Partially parsed test__m_with_empty_strings. Retrieved 3/4 statements.
# Partially parsed test__m_with_all_empty_strings. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module'
    var_1 = 'submodule'
    var_2 = 'function'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'module'
    var_1 = ''
    var_2 = 'function'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_walk_body_with_if_stmt. Retrieved 4/8 statements.
# Partially parsed test_walk_body_with_try_stmt. Retrieved 8/14 statements.
# Partially parsed test_walk_body_with_nested_if_and_try. Retrieved 10/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt'
    var_1 = [var_0]
    var_2 = module_0.walk_body(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [var_0])
    assert var_4 is True

def test_case_0():
    var_0 = 'stmt1'
    var_1 = [var_0]
    var_2 = 'stmt2'
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = 'stmt1'
    var_1 = [var_0]
    var_2 = 'stmt2'
    var_3 = [var_2]
    var_4 = 'stmt3'
    var_5 = [var_4]
    var_6 = 'stmt4'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'stmt1'
    var_1 = [var_0]
    var_2 = 'stmt2'
    var_3 = [var_2]
    var_4 = 'stmt3'
    var_5 = [var_4]
    var_6 = 'stmt4'
    var_7 = [var_6]
    var_8 = 'stmt5'
    var_9 = [var_8]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'stmt2'
    var_2 = 'stmt3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.walk_body(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [var_0, var_1, var_2])
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_globals_handles_ann_assign_with_value. Retrieved 5/10 statements.
# Partially parsed test_globals_handles_assign_with_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_handles_assign_without_type_comment. Retrieved 4/9 statements.
# Partially parsed test_globals_handles_uppercase_name. Retrieved 4/9 statements.
# Partially parsed test_globals_handles_all_special_case. Retrieved 5/13 statements.
# Partially parsed test_globals_ignores_non_name_assign. Retrieved 3/8 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'int'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.alias['root.x']
    assert var_8 == '42'
    var_9 = var_0.const['root.x']
    assert var_9 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 'test'
    var_4 = []
    var_5 = 'str'
    var_6 = 'root'
    var_7 = var_0.alias['root.y']
    assert var_7 == "'test'"
    var_8 = var_0.const['root.y']
    assert var_8 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = []
    var_3 = 3.14
    var_4 = []
    var_5 = 'root'
    var_6 = var_0.alias['root.z']
    assert var_6 == '3.14'
    var_7 = var_0.const['root.z']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'PI'
    var_2 = []
    var_3 = 3.14159
    var_4 = []
    var_5 = 'root'
    var_6 = var_0.alias['root.PI']
    assert var_6 == '3.14159'
    var_7 = var_0.const['root.PI']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'x'
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.imp['root']
    var_9 = bool(var_0.imp['root'] == {'root.x', 'root.y'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 1
    var_3 = []
    var_4 = 'root'
    var_5 = bool(not var_0.alias)
    assert var_5 is True
    var_6 = bool(not var_0.const)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = []
    var_3 = 'b'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = bool(not var_0.alias)
    assert var_8 is True
    var_9 = bool(not var_0.const)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_imports_method_import. Retrieved 6/10 statements.
# Partially parsed test_imports_method_import_with_alias. Retrieved 6/10 statements.
# Partially parsed test_imports_method_import_from. Retrieved 8/12 statements.
# Partially parsed test_imports_method_import_from_with_alias. Retrieved 8/12 statements.
# Partially parsed test_imports_method_import_from_with_level. Retrieved 8/12 statements.
# Partially parsed test_imports_method_import_from_with_level_and_alias. Retrieved 8/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = None
    var_6 = var_0.alias
    var_7 = bool(var_0.alias == {'root.module': 'module'})
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = 'alias'
    var_6 = var_0.alias
    var_7 = bool(var_0.alias == {'root.alias': 'module'})
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = 'name'
    var_6 = None
    var_7 = 0
    var_8 = var_0.alias
    var_9 = bool(var_0.alias == {'root.name': 'module.name'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = 'name'
    var_6 = 'alias'
    var_7 = 0
    var_8 = var_0.alias
    var_9 = bool(var_0.alias == {'root.alias': 'module.name'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.sub'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = 'name'
    var_6 = None
    var_7 = 1
    var_8 = var_0.alias
    var_9 = bool(var_0.alias == {'root.sub.name': 'root.module.name'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.sub'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module'
    var_5 = 'name'
    var_6 = 'alias'
    var_7 = 1
    var_8 = var_0.alias
    var_9 = bool(var_0.alias == {'root.sub.alias': 'root.module.name'})
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'example'
    var_2 = 'data'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['`test`', '`example`', '`data`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test&'
    var_1 = 'example&'
    var_2 = 'data&'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['<code>test&#38;</code>', '<code>example&#38;</code>', '<code>data&#38;</code>'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test|'
    var_1 = 'example|'
    var_2 = 'data|'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['`test&#124;`', '`example&#124;`', '`data&#124;`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = 'example&'
    var_3 = 'data|'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._defaults(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [' ', '`test`', '<code>example&#38;</code>', '`data&#124;`'])
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/10 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 5/13 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 5/13 statements.
# Partially parsed test_globals_with_uppercase_name. Retrieved 4/10 statements.
# Partially parsed test_globals_with_non_constant_value. Retrieved 5/15 statements.
# Partially parsed test_globals_with_all_assignment. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.x']
    assert var_7 == '42'
    var_8 = var_0.const['root.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'test'
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.y']
    assert var_5 == "'test'"
    var_6 = var_0.const['root.y']
    assert var_6 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'float'
    var_5 = 'root'
    var_6 = var_0.alias['root.z']
    assert var_6 == '3.14'
    var_7 = var_0.const['root.z']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arr'
    var_2 = []
    var_3 = 0
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = 'root.arr'
    var_9 = bool('root.arr' not in var_0.alias)
    assert var_9 is True
    var_10 = 'root.arr'
    var_11 = bool('root.arr' not in var_0.const)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = []
    var_5 = 'root'
    var_6 = 'root.a'
    var_7 = bool('root.a' not in var_0.alias)
    assert var_7 is True
    var_8 = 'root.b'
    var_9 = bool('root.b' not in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 100
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.CONST']
    assert var_5 == '100'
    var_6 = var_0.const['root.CONST']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'func'
    var_2 = 'len'
    var_3 = []
    var_4 = 'test'
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.func']
    assert var_7 == "len('test')"
    var_8 = var_0.const['root.func']
    assert var_8 == 'Any'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'func'
    var_3 = []
    var_4 = 'CONST'
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.imp['root']
    var_8 = bool(var_0.imp['root'] == {'root.func', 'root.CONST'})
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_simple_module. Retrieved 6/7 statements.
# Partially parsed test_parse_with_function_def. Retrieved 7/8 statements.
# Partially parsed test_parse_with_class_def. Retrieved 7/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.root)
    assert var_9 is True
    var_10 = 'test_module'
    var_11 = bool('test_module' in var_0.imp)
    assert var_11 is True
    var_12 = var_0.doc[var_1]
    var_13 = '# Module `test_module`'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os\ndef foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'os'
    var_5 = bool('os' in var_0.alias)
    assert var_5 is True
    var_6 = '_m(test_module, os)'
    var_7 = bool('_m(test_module, os)' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os import path\ndef foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'path'
    var_5 = bool('path' in var_0.alias)
    assert var_5 is True
    var_6 = '_m(test_module, path)'
    var_7 = bool('_m(test_module, path)' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_m(test_module, foo)'
    var_5 = bool('_m(test_module, foo)' in var_0.doc)
    assert var_5 is True
    var_6 = '_m(test_module, foo)'
    var_7 = var_0.doc[var_6]
    var_8 = '### foo()'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class Foo: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_m(test_module, Foo)'
    var_5 = bool('_m(test_module, Foo)' in var_0.doc)
    assert var_5 is True
    var_6 = '_m(test_module, Foo)'
    var_7 = var_0.doc[var_6]
    var_8 = '### class Foo'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "CONST = 42\n__all__ = ['foo']\ndef foo(): pass"
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_m(test_module, CONST)'
    var_5 = bool('_m(test_module, CONST)' in var_0.const)
    assert var_5 is True
    var_6 = '_m(test_module, foo)'
    var_7 = bool('_m(test_module, foo)' in var_0.imp['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '"""Module docstring"""\ndef foo(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = 'Module docstring'
    var_7 = bool('Module docstring' in var_0.docstring['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    class Inner: pass'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = '_m(test_module, Outer)'
    var_5 = bool('_m(test_module, Outer)' in var_0.doc)
    assert var_5 is True
    var_6 = '_m(test_module, Outer.Inner)'
    var_7 = bool('_m(test_module, Outer.Inner)' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_empty_elements.
# Partially parsed test_none_element. Retrieved 1/2 statements.
# Failed to parse test_non_constant_element.
# Partially parsed test_single_constant_element. Retrieved 1/7 statements.
# Partially parsed test_multiple_constant_elements_same_type. Retrieved 2/9 statements.
# Partially parsed test_multiple_constant_elements_different_types. Retrieved 2/9 statements.
# Partially parsed test_multiple_elements_with_none. Retrieved 3/10 statements.
# Partially parsed test_multiple_sequences_of_constants. Retrieved 4/14 statements.
# Partially parsed test_multiple_sequences_of_constants_mixed_types. Retrieved 4/14 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = 2
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_imports_with_node_module_none. Retrieved 4/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = None
    var_3 = 'root'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_globals_assign_with_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_ann_assign_with_annotation. Retrieved 5/12 statements.
# Partially parsed test_globals_assign_without_type_comment. Retrieved 4/10 statements.
# Partially parsed test_globals_assign_with_complex_value. Retrieved 7/18 statements.
# Partially parsed test_globals_assign_with_list_value. Retrieved 5/15 statements.
# Partially parsed test_globals_assign_with_dict_value. Retrieved 5/15 statements.
# Partially parsed test_globals_assign_with_tuple_value. Retrieved 5/15 statements.
# Partially parsed test_globals_assign_with_set_value. Retrieved 5/14 statements.
# Partially parsed test_globals_assign_with_non_constant_value. Retrieved 6/14 statements.
# Partially parsed test_globals_assign_with_non_name_target. Retrieved 7/16 statements.
# Partially parsed test_globals_assign_with_multiple_targets. Retrieved 8/18 statements.
# Partially parsed test_globals_assign_with_non_upper_case_name. Retrieved 6/13 statements.
# Partially parsed test_globals_assign_with_upper_case_name. Retrieved 4/10 statements.
# Partially parsed test_globals_assign_with_all. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = []
    var_3 = 'int'
    var_4 = module_0.Parser()
    var_5 = 'root'
    var_6 = var_4.const['root.x']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = 'str'
    var_2 = []
    var_3 = 'hello'
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.const['root.y']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 3.14
    var_2 = []
    var_3 = module_0.Parser()
    var_4 = 'root'
    var_5 = var_3.const['root.z']
    assert var_5 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'w'
    var_1 = 'complex'
    var_2 = []
    var_3 = 1
    var_4 = []
    var_5 = 2
    var_6 = []
    var_7 = []
    var_8 = module_0.Parser()
    var_9 = 'root'
    var_10 = var_8.const['root.w']
    assert var_10 == 'complex'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'lst'
    var_1 = 1
    var_2 = []
    var_3 = 2
    var_4 = []
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 'root'
    var_8 = var_6.const['root.lst']
    assert var_8 == 'list[int, int]'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'dct'
    var_1 = 'a'
    var_2 = []
    var_3 = 1
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.const['root.dct']
    assert var_7 == 'dict[str, int]'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'tpl'
    var_1 = 1
    var_2 = []
    var_3 = 2
    var_4 = []
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 'root'
    var_8 = var_6.const['root.tpl']
    assert var_8 == 'tuple[int, int]'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'st'
    var_1 = 1
    var_2 = []
    var_3 = 2
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.const['root.st']
    assert var_7 == 'set[int, int]'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = []
    var_3 = module_0.Parser()
    var_4 = 'root'
    var_5 = 'root.x'
    var_6 = 'ANY'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = []
    var_2 = 'x'
    var_3 = 42
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = 'root.obj.x'
    var_8 = 'ANY'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 42
    var_3 = []
    var_4 = module_0.Parser()
    var_5 = 'root'
    var_6 = 'root.a'
    var_7 = 'ANY'
    var_8 = 'root.b'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = []
    var_3 = module_0.Parser()
    var_4 = 'root'
    var_5 = 'root.x'
    var_6 = 'ANY'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'X'
    var_1 = 42
    var_2 = []
    var_3 = module_0.Parser()
    var_4 = 'root'
    var_5 = var_3.const['root.X']
    assert var_5 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = '__all__'
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 'root'
    var_8 = var_6.imp['root']
    var_9 = bool(var_6.imp['root'] == {'root.x', 'root.y'})
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_globals_ann_assign_with_value. Retrieved 5/12 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_assign_without_type_comment. Retrieved 4/10 statements.
# Partially parsed test_globals_assign_multiple_targets. Retrieved 5/13 statements.
# Partially parsed test_globals_non_name_target. Retrieved 5/14 statements.
# Partially parsed test_globals_non_uppercase_name. Retrieved 4/10 statements.
# Partially parsed test_globals_uppercase_name. Retrieved 4/10 statements.
# Partially parsed test_globals_all_special_case. Retrieved 5/15 statements.
# Partially parsed test_globals_all_non_list_tuple. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.x']
    assert var_7 == '42'
    var_8 = var_0.const['root.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'test'
    var_3 = []
    var_4 = 'str'
    var_5 = 'root'
    var_6 = var_0.alias['root.y']
    assert var_6 == "'test'"
    var_7 = var_0.const['root.y']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.z']
    assert var_5 == '3.14'
    var_6 = var_0.const['root.z']
    assert var_6 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = []
    var_5 = 'root'
    var_6 = 'root.a'
    var_7 = bool('root.a' not in var_0.alias)
    assert var_7 is True
    var_8 = 'root.b'
    var_9 = bool('root.b' not in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'lst'
    var_2 = []
    var_3 = 0
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = 'root.lst'
    var_9 = bool('root.lst' not in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = 1
    var_3 = []
    var_4 = 'root'
    var_5 = 'root.var'
    var_6 = bool('root.var' in var_0.alias)
    assert var_6 is True
    var_7 = 'root.var'
    var_8 = bool('root.var' not in var_0.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 1
    var_3 = []
    var_4 = 'root'
    var_5 = 'root.CONST'
    var_6 = bool('root.CONST' in var_0.alias)
    assert var_6 is True
    var_7 = 'root.CONST'
    var_8 = bool('root.CONST' in var_0.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'x'
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'root.x'
    var_9 = bool('root.x' in var_0.imp['root'])
    assert var_9 is True
    var_10 = 'root.y'
    var_11 = bool('root.y' in var_0.imp['root'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 1
    var_3 = []
    var_4 = 'root'
    var_5 = bool(not var_0.imp['root'])
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__attr_with_single_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_with_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test__attr_with_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test__attr_with_nonexistent_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test__attr_with_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'inner_value'
    var_1 = 'outer_attr.inner_attr'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'outer_attr.nonexistent'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'attr'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #14
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
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 8/19 statements.
# Partially parsed test_class_api_with_enums. Retrieved 8/19 statements.
# Partially parsed test_class_api_without_bases_and_members. Retrieved 8/12 statements.
# Partially parsed test_class_api_with_delete_statement. Retrieved 8/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = 'type'
    var_4 = 'root.Class'
    var_5 = 'Base'
    var_6 = []
    var_7 = 'attr'
    var_8 = []
    var_9 = 'int'
    var_10 = []
    var_11 = 'root.Class'
    var_12 = bool('root.Class' in var_0.doc)
    assert var_12 is True
    var_13 = 'Bases'
    var_14 = bool('Bases' in var_0.doc['root.Class'])
    assert var_14 is True
    var_15 = 'Members'
    var_16 = bool('Members' in var_0.doc['root.Class'])
    assert var_16 is True
    var_17 = 'attr'
    var_18 = bool('attr' in var_0.doc['root.Class'])
    assert var_18 is True
    var_19 = 'int'
    var_20 = bool('int' in var_0.doc['root.Class'])
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = 'type'
    var_4 = 'root.Class'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = 'ENUM_VALUE'
    var_8 = []
    var_9 = 'int'
    var_10 = []
    var_11 = 'root.Class'
    var_12 = bool('root.Class' in var_0.doc)
    assert var_12 is True
    var_13 = 'Enums'
    var_14 = bool('Enums' in var_0.doc['root.Class'])
    assert var_14 is True
    var_15 = 'ENUM_VALUE'
    var_16 = bool('ENUM_VALUE' in var_0.doc['root.Class'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = 'type'
    var_4 = 'root.Class'
    var_5 = []
    var_6 = []
    var_7 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_8 = 'root.Class'
    var_9 = bool('root.Class' in var_0.doc)
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' not in var_0.doc['root.Class'])
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' not in var_0.doc['root.Class'])
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' not in var_0.doc['root.Class'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 0
    var_3 = 'type'
    var_4 = 'root.Class'
    var_5 = []
    var_6 = 'attr'
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = []
    var_11 = 'root.Class'
    var_12 = bool('root.Class' in var_0.doc)
    assert var_12 is True
    var_13 = 'Members'
    var_14 = bool('Members' not in var_0.doc['root.Class'])
    assert var_14 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 6/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 7/15 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 7/19 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 7/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'Base'
    var_5 = []
    var_6 = []
    var_7 = 'Bases'
    var_8 = bool('Bases' in var_0.doc['root.Class'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.EnumClass'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'enum.Enum'
    var_5 = []
    var_6 = 'ENUM_VALUE'
    var_7 = 1
    var_8 = []
    var_9 = 'Enums'
    var_10 = bool('Enums' in var_0.doc['root.EnumClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.ClassWithMembers'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'member'
    var_6 = 'int'
    var_7 = []
    var_8 = 'Members'
    var_9 = bool('Members' in var_0.doc['root.ClassWithMembers'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.ClassWithDeletedMember'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'member'
    var_6 = 'int'
    var_7 = []
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc['root.ClassWithDeletedMember'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.ClassWithPrivateMember'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = '_private'
    var_6 = 'int'
    var_7 = []
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc['root.ClassWithPrivateMember'])
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func_name'
    var_12 = False
    var_13 = '| x | y | / | return |'
    var_14 = bool('| x | y | / | return |' in var_0.doc['func_name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|:---:|:---:|'
    var_16 = bool('|:---:|:---:|:---:|:---:|' in var_0.doc['func_name'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = 'b'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 1
    var_10 = []
    var_11 = 'root'
    var_12 = 'func_name'
    var_13 = False
    var_14 = '| a | b | return |'
    var_15 = bool('| a | b | return |' in var_0.doc['func_name'])
    assert var_15 is True
    var_16 = '|:---:|:---:|:---:|'
    var_17 = bool('|:---:|:---:|:---:|' in var_0.doc['func_name'])
    assert var_17 is True
    var_18 = '|  | 1 |  |'
    var_19 = bool('|  | 1 |  |' in var_0.doc['func_name'])
    assert var_19 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'args'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'func_name'
    var_11 = False
    var_12 = '| *args | return |'
    var_13 = bool('| *args | return |' in var_0.doc['func_name'])
    assert var_13 is True
    var_14 = '|:---:|:---:|'
    var_15 = bool('|:---:|:---:|' in var_0.doc['func_name'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'x'
    var_5 = []
    var_6 = 'y'
    var_7 = []
    var_8 = 2
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'func_name'
    var_13 = False
    var_14 = '| * | x | y | return |'
    var_15 = bool('| * | x | y | return |' in var_0.doc['func_name'])
    assert var_15 is True
    var_16 = '|:---:|:---:|:---:|:---:|'
    var_17 = bool('|:---:|:---:|:---:|:---:|' in var_0.doc['func_name'])
    assert var_17 is True
    var_18 = '|  |  | 2 |  |'
    var_19 = bool('|  |  | 2 |  |' in var_0.doc['func_name'])
    assert var_19 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'func_name'
    var_11 = False
    var_12 = '| **kwargs | return |'
    var_13 = bool('| **kwargs | return |' in var_0.doc['func_name'])
    assert var_13 is True
    var_14 = '|:---:|:---:|'
    var_15 = bool('|:---:|:---:|' in var_0.doc['func_name'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func_name'
    var_12 = True
    var_13 = False
    var_14 = '| Self | x | return |'
    var_15 = bool('| Self | x | return |' in var_0.doc['func_name'])
    assert var_15 is True
    var_16 = '|:---:|:---:|:---:|'
    var_17 = bool('|:---:|:---:|:---:|' in var_0.doc['func_name'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func_name'
    var_12 = True
    var_13 = '| type[Self] | x | return |'
    var_14 = bool('| type[Self] | x | return |' in var_0.doc['func_name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|:---:|'
    var_16 = bool('|:---:|:---:|:---:|' in var_0.doc['func_name'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func_name'
    var_10 = 'int'
    var_11 = []
    var_12 = False
    var_13 = '| x | return |'
    var_14 = bool('| x | return |' in var_0.doc['func_name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|'
    var_16 = bool('|:---:|:---:|' in var_0.doc['func_name'])
    assert var_16 is True
    var_17 = '|  | int |'
    var_18 = bool('|  | int |' in var_0.doc['func_name'])
    assert var_18 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_tuple_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_mixed_tuple. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_floats. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_set_of_strings. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_dict_of_int_to_str. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_builtin_func_call. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unknown_node. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1.1
    var_1 = []
    var_2 = 2.2
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'y'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__attr_single_level. Retrieved 2/5 statements.
# Partially parsed test__attr_nested_level. Retrieved 2/7 statements.
# Partially parsed test__attr_non_existent_single_level. Retrieved 1/5 statements.
# Partially parsed test__attr_non_existent_nested_level. Retrieved 1/7 statements.
# Partially parsed test__attr_non_existent_intermediate_level. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'inner_value'
    var_1 = 'nested.inner_attr'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'nested.nonexistent'

def test_case_0():
    var_0 = 'nonexistent.intermediate'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_visit_Subscript_Union. Retrieved 7/26 statements.
# Partially parsed test_visit_Subscript_Optional. Retrieved 6/20 statements.
# Partially parsed test_visit_Subscript_PEP585. Retrieved 6/18 statements.
# Partially parsed test_visit_Subscript_other. Retrieved 7/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Union'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = 'str'
    var_9 = []
    var_10 = []
    var_11 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Optional'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'List'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'Dict'
    var_5 = []
    var_6 = 'str'
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = []
    var_11 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_api_parsing_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_parsing_async_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_parsing_class_def. Retrieved 7/9 statements.
# Partially parsed test_api_parsing_with_decorators. Retrieved 11/17 statements.
# Partially parsed test_api_parsing_with_prefix. Retrieved 12/15 statements.
# Partially parsed test_api_parsing_with_docstring. Retrieved 11/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = var_0.doc['test_module.test_func']
    assert var_12 == '### test_func()\n\n*Full name:* `test_module.test_func`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = var_0.doc['test_module.test_async_func']
    assert var_12 == '### async test_async_func()\n\n*Full name:* `test_module.test_async_func`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = var_0.doc['test_module.TestClass']
    assert var_8 == '### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'decorated_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'decorator'
    var_12 = []
    var_13 = '| Decorators |\n|:---:|\n| `@decorator` |'
    var_14 = bool('| Decorators |\n|:---:|\n| `@decorator` |' in var_0.doc['test_module.decorated_func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'method'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'TestClass'
    var_13 = var_0.doc['test_module.TestClass.method']
    assert var_13 == '#### method()\n\n*Full name:* `test_module.TestClass.method`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'doc_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'This is a docstring'
    var_11 = []
    var_12 = []
    var_13 = 'This is a docstring'
    var_14 = bool('This is a docstring' in var_0.docstring['test_module.doc_func'])
    assert var_14 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_e_type_empty_elements.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_imports_with_node_level. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'root'
    var_5 = 'root.name'
    var_6 = bool('root.name' in var_0.alias)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_default_values. Retrieved 10/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func_name'
    var_10 = False
    var_11 = '| x |'
    var_12 = bool('| x |' in var_0.doc['func_name'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'args'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'func_name'
    var_11 = False
    var_12 = '| *args |'
    var_13 = bool('| *args |' in var_0.doc['func_name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'func_name'
    var_11 = False
    var_12 = '| **kwargs |'
    var_13 = bool('| **kwargs |' in var_0.doc['func_name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func_name'
    var_10 = True
    var_11 = False
    var_12 = '| Self |'
    var_13 = bool('| Self |' in var_0.doc['func_name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func_name'
    var_10 = True
    var_11 = '| type[Self] |'
    var_12 = bool('| type[Self] |' in var_0.doc['func_name'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func_name'
    var_10 = 'str'
    var_11 = []
    var_12 = False
    var_13 = '| str |'
    var_14 = bool('| str |' in var_0.doc['func_name'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 1
    var_8 = []
    var_9 = 'root'
    var_10 = 'func_name'
    var_11 = False
    var_12 = '| 1 |'
    var_13 = bool('| 1 |' in var_0.doc['func_name'])
    assert var_13 is True



# Parsed testcases at query #26
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A(B, C): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = bool('test_module.A' in var_0.doc)
    assert var_5 is True
    var_6 = 'Bases'
    var_7 = bool('Bases' in var_0.doc['test_module.A'])
    assert var_7 is True
    var_8 = 'B'
    var_9 = bool('B' in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'C'
    var_11 = bool('C' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A(enum.Enum): X = 1; Y = 2'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = bool('test_module.A' in var_0.doc)
    assert var_5 is True
    var_6 = 'Enums'
    var_7 = bool('Enums' in var_0.doc['test_module.A'])
    assert var_7 is True
    var_8 = 'X'
    var_9 = bool('X' in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'Y'
    var_11 = bool('Y' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: x: int; y: str'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = bool('test_module.A' in var_0.doc)
    assert var_5 is True
    var_6 = 'Members'
    var_7 = bool('Members' in var_0.doc['test_module.A'])
    assert var_7 is True
    var_8 = 'x'
    var_9 = bool('x' in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'int'
    var_11 = bool('int' in var_0.doc['test_module.A'])
    assert var_11 is True
    var_12 = 'y'
    var_13 = bool('y' in var_0.doc['test_module.A'])
    assert var_13 is True
    var_14 = 'str'
    var_15 = bool('str' in var_0.doc['test_module.A'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: x: int; del x'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = bool('test_module.A' in var_0.doc)
    assert var_5 is True
    var_6 = 'Members'
    var_7 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: _x: int; y: str'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = bool('test_module.A' in var_0.doc)
    assert var_5 is True
    var_6 = 'Members'
    var_7 = bool('Members' in var_0.doc['test_module.A'])
    assert var_7 is True
    var_8 = '_x'
    var_9 = bool('_x' not in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'y'
    var_11 = bool('y' in var_0.doc['test_module.A'])
    assert var_11 is True



# Parsed testcases at query #27
#--------------------------

# Failed to parse test__e_type_empty_elements.
# Partially parsed test__e_type_empty_sequence. Retrieved 1/2 statements.
# Failed to parse test__e_type_non_constant_element.
# Partially parsed test__e_type_single_constant_element. Retrieved 1/4 statements.
# Partially parsed test__e_type_multiple_constant_elements_same_type. Retrieved 2/6 statements.
# Partially parsed test__e_type_multiple_constant_elements_different_types. Retrieved 2/6 statements.
# Partially parsed test__e_type_multiple_sequences_same_type. Retrieved 3/9 statements.
# Partially parsed test__e_type_multiple_sequences_different_types. Retrieved 3/9 statements.
# Partially parsed test__e_type_mixed_sequences. Retrieved 3/9 statements.
# Partially parsed test__e_type_none_element. Retrieved 2/3 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = 24
    var_3 = [var_2]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = '24'
    var_3 = [var_2]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = 24
    var_3 = [var_2]
    var_4 = 99
    var_5 = [var_4]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = '24'
    var_3 = [var_2]
    var_4 = 99
    var_5 = [var_4]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = 24
    var_3 = [var_2]
    var_4 = '99'
    var_5 = [var_4]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_class_api_with_non_annassign_node. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_name'
    var_3 = []
    var_4 = 'x'
    var_5 = 42
    var_6 = []
    var_7 = 'Enums'
    var_8 = bool('Enums' not in var_0.doc[var_2])
    assert var_8 is True
    var_9 = 'Members'
    var_10 = bool('Members' not in var_0.doc[var_2])
    assert var_10 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_name'
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = 'y'
    var_7 = []
    var_8 = 10
    var_9 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_visit_Attribute_with_typing_prefix. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_without_typing_prefix. Retrieved 5/10 statements.
# Partially parsed test_visit_Attribute_with_non_name_value. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'module'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 15/23 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 18/30 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 18/30 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 14/20 statements.
# Partially parsed test_func_api_with_self_param. Retrieved 13/18 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 13/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arg1'
    var_2 = None
    var_3 = []
    var_4 = 'arg2'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.doc[var_11]
    var_14 = var_0.doc[var_11]
    var_15 = var_0.doc[var_11]
    var_16 = '/'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'arg1'
    var_3 = None
    var_4 = []
    var_5 = 'arg2'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 1
    var_10 = []
    var_11 = 2
    var_12 = []
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = var_0.doc[var_14]
    var_17 = var_0.doc[var_14]
    var_18 = var_0.doc[var_14]
    var_19 = '1'
    var_20 = var_0.doc[var_14]
    var_21 = '2'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'args'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False
    var_12 = var_0.doc[var_10]
    var_13 = '*args'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'kw1'
    var_5 = []
    var_6 = 'kw2'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 2
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = var_0.doc[var_14]
    var_17 = var_0.doc[var_14]
    var_18 = var_0.doc[var_14]
    var_19 = '1'
    var_20 = var_0.doc[var_14]
    var_21 = '2'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False
    var_12 = var_0.doc[var_10]
    var_13 = '**kwargs'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = 'int'
    var_11 = []
    var_12 = False
    var_13 = var_0.doc[var_9]
    var_14 = 'return'
    var_15 = var_0.doc[var_9]

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True
    var_11 = False
    var_12 = var_0.doc[var_9]
    var_13 = 'Self'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = 'Type'
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = True
    var_12 = var_0.doc[var_10]
    var_13 = 'type[Self]'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_posonlyargs_condition_evaluates_to_true. Retrieved 12/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'arg1'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = module_0.Parser()
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = var_7.doc[var_9]
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_23_evaluates_to_false. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'root'
    var_7 = 'root.x'
    var_8 = 'ANY'



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_func_api_with_defaults_and_decorators. Retrieved 13/22 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 15/20 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 15/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'example'
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = 'b'
    var_6 = []
    var_7 = [var_3, var_3]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'staticmethod'
    var_12 = []
    var_13 = 'root'
    var_14 = 'root.example'
    var_15 = False
    var_16 = var_0.doc['root.example']
    assert var_16 == '### example()\n\n*Full name:* `root.example`\n\n| a | b |\n|:---:|:---:|\n|  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'example'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'args'
    var_8 = None
    var_9 = []
    var_10 = 'kwargs'
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = 'root'
    var_15 = 'root.example'
    var_16 = False
    var_17 = var_0.doc['root.example']
    assert var_17 == '### example()\n\n*Full name:* `root.example`\n\n| * | args | ** | kwargs | return |\n|:---:|:---:|:---:|:---:|:---:|\n|  |  |  |  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'example'
    var_2 = []
    var_3 = 'a'
    var_4 = 'int'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = []
    var_12 = 'str'
    var_13 = []
    var_14 = 'root'
    var_15 = 'root.example'
    var_16 = []
    var_17 = False
    var_18 = var_0.doc['root.example']
    assert var_18 == '### example()\n\n*Full name:* `root.example`\n\n| a | return |\n|:---:|:---:|\n| int | str |\n\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_api_method_for_function_with_decorators. Retrieved 6/13 statements.
# Partially parsed test_api_method_for_async_function. Retrieved 4/12 statements.
# Partially parsed test_api_method_for_class. Retrieved 4/12 statements.
# Partially parsed test_api_method_for_function_with_self_and_classmethod. Retrieved 6/13 statements.
# Partially parsed test_api_method_for_function_with_self_and_staticmethod. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = '@decorator1'
    var_3 = '@decorator2'
    var_4 = [var_2, var_3]
    var_5 = 'root'
    var_6 = var_0.doc['root.test_function']
    assert var_6 == '### test_function()\n\n*Full name:* `root.test_function`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = []
    var_3 = 'root'
    var_4 = var_0.doc['root.test_function']
    assert var_4 == '### async test_function()\n\n*Full name:* `root.test_function`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'root'
    var_4 = var_0.doc['root.TestClass']
    assert var_4 == '### class TestClass\n\n*Full name:* `root.TestClass`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = '@classmethod'
    var_3 = [var_2]
    var_4 = 'root'
    var_5 = 'TestClass'
    var_6 = var_0.doc['root.TestClass.test_function']
    assert var_6 == '#### test_function()\n\n*Full name:* `root.TestClass.test_function`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = '@staticmethod'
    var_3 = [var_2]
    var_4 = 'root'
    var_5 = 'TestClass'
    var_6 = var_0.doc['root.TestClass.test_function']
    assert var_6 == '#### test_function()\n\n*Full name:* `root.TestClass.test_function`\n\n'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_api_parses_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_parses_async_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_parses_class_def. Retrieved 7/9 statements.
# Partially parsed test_api_parses_class_def_with_prefix. Retrieved 8/10 statements.
# Partially parsed test_api_parses_function_def_with_decorators. Retrieved 11/17 statements.
# Partially parsed test_api_parses_class_def_with_bases. Retrieved 7/12 statements.
# Partially parsed test_api_parses_class_def_with_docstring. Retrieved 7/12 statements.
# Partially parsed test_api_parses_function_def_with_docstring. Retrieved 11/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'test_func()'
    var_13 = bool('test_func()' in var_0.doc['test_module.test_func'])
    assert var_13 is True
    var_14 = '*Full name*: `test_module.test_func`'
    var_15 = bool('*Full name*: `test_module.test_func`' in var_0.doc['test_module.test_func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'async test_async_func()'
    var_13 = bool('async test_async_func()' in var_0.doc['test_module.test_async_func'])
    assert var_13 is True
    var_14 = '*Full name*: `test_module.test_async_func`'
    var_15 = bool('*Full name*: `test_module.test_async_func`' in var_0.doc['test_module.test_async_func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'class TestClass'
    var_9 = bool('class TestClass' in var_0.doc['test_module.TestClass'])
    assert var_9 is True
    var_10 = '*Full name*: `test_module.TestClass`'
    var_11 = bool('*Full name*: `test_module.TestClass`' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'prefix'
    var_9 = 'class prefix.TestClass'
    var_10 = bool('class prefix.TestClass' in var_0.doc['test_module.prefix.TestClass'])
    assert var_10 is True
    var_11 = '*Full name*: `test_module.prefix.TestClass`'
    var_12 = bool('*Full name*: `test_module.prefix.TestClass`' in var_0.doc['test_module.prefix.TestClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'decorator'
    var_3 = []
    var_4 = 'test_func'
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = '@decorator'
    var_14 = bool('@decorator' in var_0.doc['test_module.test_func'])
    assert var_14 is True
    var_15 = 'Decorators'
    var_16 = bool('Decorators' in var_0.doc['test_module.test_func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'BaseClass'
    var_3 = []
    var_4 = 'TestClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_9 is True
    var_10 = 'BaseClass'
    var_11 = bool('BaseClass' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'This is a test class'
    var_3 = 'TestClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = bool(var_2 in var_0.docstring['test_module.TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'This is a test function'
    var_3 = 'test_func'
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = bool(var_2 in var_0.docstring['test_module.test_func'])
    assert var_13 is True



# Parsed testcases at query #4
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello, World!')"
    var_1 = module_0.doctest(var_0)
    var_2 = "```python\n>>> print('Hello, World!')\n```"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello, World!')\n>>> print('Goodbye, World!')"
    var_1 = module_0.doctest(var_0)
    var_2 = "```python\n>>> print('Hello, World!')\n>>> print('Goodbye, World!')\n```"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "Some text\n>>> print('Hello, World!')\nMore text"
    var_1 = module_0.doctest(var_0)
    var_2 = "Some text\n```python\n>>> print('Hello, World!')\n```\nMore text"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Some text\nMore text'
    var_1 = module_0.doctest(var_0)
    var_2 = 'Some text\nMore text'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "Some text\n>>> print('Hello, World!')\nMore text\n>>> print('Goodbye, World!')"
    var_1 = module_0.doctest(var_0)
    var_2 = "Some text\n```python\n>>> print('Hello, World!')\n```\nMore text\n```python\n>>> print('Goodbye, World!')\n```"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "Some text\n>>> print('Hello, World!')"
    var_1 = module_0.doctest(var_0)
    var_2 = "Some text\n```python\n>>> print('Hello, World!')\n```"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello, World!')\nSome text"
    var_1 = module_0.doctest(var_0)
    var_2 = "```python\n>>> print('Hello, World!')\n```\nSome text"
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_api_with_classdef_node_and_empty_decorators. Retrieved 8/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = ''



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.os': 'os'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os as operating_system'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.operating_system': 'os'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from sys import path'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.path': 'sys.path'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from sys import path as sys_path'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.sys_path': 'sys.path'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from os.path import join'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.join': 'os.path.join'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'from ..sub import func'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test.module.func': 'test.sub.func'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os, sys as system'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.os': 'os', 'test_module.system': 'sys'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'from sys import path, argv as arguments'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'test_module.path': 'sys.path', 'test_module.arguments': 'sys.argv'})
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_compile_without_toc. Retrieved 7/13 statements.
# Partially parsed test_compile_with_toc. Retrieved 8/14 statements.
# Partially parsed test_compile_with_constants. Retrieved 10/16 statements.
# Partially parsed test_compile_with_missing_docstring. Retrieved 6/12 statements.
# Partially parsed test_compile_with_non_public_name. Retrieved 9/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module_name'
    var_3 = '## Module `module_name`\n\n'
    var_4 = 'Docstring for module_name'
    var_5 = set()
    var_6 = var_1.compile()
    assert var_6 == '## Module `module_name`\n\nDocstring for module_name\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module_name'
    var_3 = '## Module `module_name`\n\n'
    var_4 = 'Docstring for module_name'
    var_5 = 0
    var_6 = set()
    var_7 = var_1.compile()
    assert var_7 == '**Table of contents:**\n+ [module_name](#module-name)\n\n## Module `module_name`\n\nDocstring for module_name\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module_name'
    var_3 = '## Module `module_name`\n\n'
    var_4 = 'Docstring for module_name'
    var_5 = 'module_name.CONST'
    var_6 = 'int'
    var_7 = 1
    var_8 = set()
    var_9 = var_1.compile()
    assert var_9 == '## Module `module_name`\n\n| Constants | Type |\n|-----------|------|\n| CONST     | int  |\n\nDocstring for module_name\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module_name'
    var_3 = '## Module `module_name`\n\n'
    var_4 = set()
    var_5 = var_1.compile()
    assert var_5 == '## Module `module_name`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module_name._private'
    var_3 = '## Module `module_name._private`\n\n'
    var_4 = 'Docstring for module_name._private'
    var_5 = 1
    var_6 = 'module_name'
    var_7 = set()
    var_8 = var_1.compile()
    assert var_8 == ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_docstring_with_valid_module. Retrieved 3/9 statements.
# Partially parsed test_load_docstring_with_none_docstring. Retrieved 3/8 statements.
# Partially parsed test_load_docstring_with_nested_attr. Retrieved 4/11 statements.
# Partially parsed test_load_docstring_with_non_existing_attr. Retrieved 2/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = module_0.Parser()
    var_3 = var_2.docstring['test_module.test_func']
    assert var_3 == '```python\nTest docstring\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = module_0.Parser()
    var_3 = 'test_module.test_func'
    var_4 = bool('test_module.test_func' not in var_2.docstring)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'sub'
    var_2 = None
    var_3 = module_0.Parser()
    var_4 = var_3.docstring['test_module.sub.test_func']
    assert var_4 == '```python\nNested docstring\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.non_existing'
    var_3 = bool('test_module.non_existing' not in var_1.docstring)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_public_returns_true_for_public_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_false_for_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_true_for_name_in_all_list. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_false_for_name_not_in_all_list. Retrieved 4/6 statements.
# Partially parsed test_is_public_returns_true_for_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_returns_true_for_magic_name. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root._name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.other'
    var_2 = 'root.name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.__name__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__attr_single_level. Retrieved 2/4 statements.
# Partially parsed test__attr_nested_level. Retrieved 2/8 statements.
# Partially parsed test__attr_nonexistent_single_level. Retrieved 2/4 statements.
# Partially parsed test__attr_nonexistent_nested_level. Retrieved 2/8 statements.
# Partially parsed test__attr_nonexistent_intermediate_level. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'value'

def test_case_0():
    var_0 = 42
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 42
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 42
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 42
    var_1 = 'nonexistent.value'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_walk_body_with_single_node.
# Failed to parse test_walk_body_with_if_node.
# Partially parsed test_walk_body_with_try_node. Retrieved 1/23 statements.
# Partially parsed test_walk_body_with_mixed_nodes. Retrieved 1/41 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 0



# Parsed testcases at query #12
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = 'B'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = var_7.value
    var_9 = [var_8]
    var_10 = 'x: int'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_4]
    var_13 = 'y = 1'
    var_14 = module_1.parse(var_13)
    var_15 = var_14.body[var_4]
    var_16 = [var_12, var_15]
    var_17 = 'test_module.A'
    var_18 = var_0.class_api(var_1, var_17, var_9, var_16)
    var_19 = 'Bases'
    var_20 = bool('Bases' in var_0.doc['test_module.A'])
    assert var_20 is True
    var_21 = 'Members'
    var_22 = bool('Members' in var_0.doc['test_module.A'])
    assert var_22 is True
    var_23 = 'x'
    var_24 = bool('x' in var_0.doc['test_module.A'])
    assert var_24 is True
    var_25 = 'int'
    var_26 = bool('int' in var_0.doc['test_module.A'])
    assert var_26 is True
    var_27 = 'y'
    var_28 = bool('y' in var_0.doc['test_module.A'])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A(enum.Enum): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = 'enum.Enum'
    var_6 = module_1.parse(var_5)
    var_7 = var_6.body[var_4]
    var_8 = var_7.value
    var_9 = [var_8]
    var_10 = 'X = 1'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_4]
    var_13 = 'Y = 2'
    var_14 = module_1.parse(var_13)
    var_15 = var_14.body[var_4]
    var_16 = [var_12, var_15]
    var_17 = 'test_module.A'
    var_18 = var_0.class_api(var_1, var_17, var_9, var_16)
    var_19 = 'Enums'
    var_20 = bool('Enums' in var_0.doc['test_module.A'])
    assert var_20 is True
    var_21 = 'X'
    var_22 = bool('X' in var_0.doc['test_module.A'])
    assert var_22 is True
    var_23 = 'Y'
    var_24 = bool('Y' in var_0.doc['test_module.A'])
    assert var_24 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = []
    var_5 = []
    var_6 = 'test_module.A'
    var_7 = var_0.class_api(var_1, var_6, var_4, var_5)
    var_8 = 'Bases'
    var_9 = bool('Bases' not in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' not in var_0.doc['test_module.A'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = []
    var_5 = 0
    var_6 = '_x: int'
    var_7 = module_1.parse(var_6)
    var_8 = var_7.body[var_5]
    var_9 = '__y = 1'
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body[var_5]
    var_12 = [var_8, var_11]
    var_13 = 'test_module.A'
    var_14 = var_0.class_api(var_1, var_13, var_4, var_12)
    var_15 = 'Members'
    var_16 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = []
    var_5 = 0
    var_6 = 'x = 1'
    var_7 = module_1.parse(var_6)
    var_8 = var_7.body[var_5]
    var_9 = 'del x'
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body[var_5]
    var_12 = [var_8, var_11]
    var_13 = 'test_module.A'
    var_14 = var_0.class_api(var_1, var_13, var_4, var_12)
    var_15 = 'Members'
    var_16 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_16 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_docstring_with_non_none_doc. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.func'
    var_2 = 'test doc'
    var_3 = 'test_module'
    var_4 = 'test docstring'
    var_5 = module_0.doctest(var_4)
    var_6 = var_0.docstring['test_module.func']
    var_7 = bool(var_0.docstring['test_module.func'] == var_5)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'empty_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.doc
    var_5 = bool(var_0.doc == {'empty_module': '## Module `empty_module`\n\n'})
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level == {'empty_module': 0})
    assert var_7 is True
    var_8 = set()
    var_9 = {var_1: var_8}
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == var_9)
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {'empty_module': 'empty_module'})
    assert var_13 is True
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nimport sys'
    var_2 = 'import_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc
    var_5 = bool(var_0.doc == {'import_module': '## Module `import_module`\n\n'})
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level == {'import_module': 0})
    assert var_7 is True
    var_8 = set()
    var_9 = {var_2: var_8}
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == var_9)
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {'import_module': 'import_module'})
    assert var_13 is True
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo():\n    pass'
    var_2 = 'func_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc
    var_5 = bool(var_0.doc == {'func_module': '## Module `func_module`\n\n', 'func_module.foo': '### foo()\n\n*Full name:* `func_module.foo`\n\n'})
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level == {'func_module': 0, 'func_module.foo': 0})
    assert var_7 is True
    var_8 = set()
    var_9 = {var_2: var_8}
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == var_9)
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {'func_module': 'func_module', 'func_module.foo': 'func_module'})
    assert var_13 is True
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Bar:\n    pass'
    var_2 = 'class_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc
    var_5 = bool(var_0.doc == {'class_module': '## Module `class_module`\n\n', 'class_module.Bar': '### class Bar\n\n*Full name:* `class_module.Bar`\n\n'})
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level == {'class_module': 0, 'class_module.Bar': 0})
    assert var_7 is True
    var_8 = set()
    var_9 = {var_2: var_8}
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == var_9)
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {'class_module': 'class_module', 'class_module.Bar': 'class_module'})
    assert var_13 is True
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {})
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Module docstring"""'
    var_2 = 'doc_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc
    var_5 = bool(var_0.doc == {'doc_module': '## Module `doc_module`\n\n'})
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level == {'doc_module': 0})
    assert var_7 is True
    var_8 = set()
    var_9 = {var_2: var_8}
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == var_9)
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {'doc_module': 'doc_module'})
    assert var_13 is True
    var_14 = var_0.docstring
    var_15 = bool(var_0.docstring == {'doc_module': '"""Module docstring"""'})
    assert var_15 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_walk_body_yields_non_control_flow_nodes. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 1
    var_3 = 'cond'
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = 2
    var_8 = []
    var_9 = 'z'
    var_10 = []
    var_11 = 3
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_magic_and_has_docstring. Retrieved 4/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__magic__'
    var_2 = 'docstring'
    var_3 = var_0.compile()
    var_4 = bool(var_3 != '')
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 6/10 statements.
# Partially parsed test_class_api_with_enums. Retrieved 7/15 statements.
# Partially parsed test_class_api_with_members. Retrieved 9/17 statements.
# Partially parsed test_class_api_with_delete_statement. Retrieved 7/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = var_0.doc['name']
    assert var_6 == '### class name\n\n*Full name:* `name`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = 'base1'
    var_4 = [var_3]
    var_5 = 'base2'
    var_6 = [var_5]
    var_7 = []
    var_8 = var_0.doc['name']
    assert var_8 == '### class name\n\n*Full name:* `name`\n\n| Bases |\n|:---:|\n| `base1` |\n| `base2` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = 'enum.Enum'
    var_4 = [var_3]
    var_5 = 'attr1'
    var_6 = [var_5]
    var_7 = None
    var_8 = 'attr2'
    var_9 = [var_8]
    var_10 = var_0.doc['name']
    assert var_10 == '### class name\n\n*Full name:* `name`\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n\n| Enums |\n|:---:|\n| attr1 |\n| attr2 |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = 'attr1'
    var_5 = [var_4]
    var_6 = None
    var_7 = 'type1'
    var_8 = [var_7]
    var_9 = 'attr2'
    var_10 = [var_9]
    var_11 = 'type2'
    var_12 = [var_11]
    var_13 = var_0.doc['name']
    assert var_13 == '### class name\n\n*Full name:* `name`\n\n| Members | Type |\n|:---:|:---:|\n| `attr1` | `type1` |\n| `attr2` | `type2` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = 'attr1'
    var_5 = [var_4]
    var_6 = None
    var_7 = 'type1'
    var_8 = [var_7]
    var_9 = [var_4]
    var_10 = var_0.doc['name']
    assert var_10 == '### class name\n\n*Full name:* `name`\n\n'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 6/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'y'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 1
    var_8 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 11/17 statements.
# Partially parsed test_compile_without_toc. Retrieved 11/17 statements.
# Partially parsed test_compile_with_constants. Retrieved 9/15 statements.
# Partially parsed test_compile_with_missing_docstring_warning. Retrieved 10/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.func'
    var_4 = '### Module `module`\n\n'
    var_5 = '#### func()\n\n*Full name:* `module.func`\n\n'
    var_6 = 'Function docstring'
    var_7 = 0
    var_8 = set()
    var_9 = var_1.compile()
    var_10 = '**Table of contents:**\n    + [`module`](#module)\n        + [`module.func`](#module-func)\n\n### Module `module`\n\n\n#### func()\n\n*Full name:* `module.func`\n\nFunction docstring\n'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.func'
    var_4 = '### Module `module`\n\n'
    var_5 = '#### func()\n\n*Full name:* `module.func`\n\n'
    var_6 = 'Function docstring'
    var_7 = 1
    var_8 = set()
    var_9 = var_1.compile()
    var_10 = '### Module `module`\n\n\n#### func()\n\n*Full name:* `module.func`\n\nFunction docstring\n'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '### Module `module`\n\n'
    var_4 = set()
    var_5 = 'module.CONST'
    var_6 = 'int'
    var_7 = var_1.compile()
    var_8 = '### Module `module`\n\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.__magic__'
    var_4 = '### Module `module`\n\n'
    var_5 = '#### __magic__()\n\n*Full name:* `module.__magic__`\n\n'
    var_6 = 1
    var_7 = set()
    var_8 = var_1.compile()
    var_9 = '### Module `module`\n\n\n#### __magic__()\n\n*Full name:* `module.__magic__`\n\n'
    var_10 = bool(var_8 == var_9)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_api_method_creates_function_doc. Retrieved 11/14 statements.
# Partially parsed test_api_method_creates_async_function_doc. Retrieved 11/14 statements.
# Partially parsed test_api_method_creates_class_doc. Retrieved 7/9 statements.
# Partially parsed test_api_method_includes_decorators. Retrieved 11/17 statements.
# Partially parsed test_api_method_includes_docstring. Retrieved 11/17 statements.
# Partially parsed test_api_method_nested_class. Retrieved 16/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'test_func()'
    var_13 = bool('test_func()' in var_0.doc['test_module.test_func'])
    assert var_13 is True
    var_14 = '*Full name:* `test_module.test_func`'
    var_15 = bool('*Full name:* `test_module.test_func`' in var_0.doc['test_module.test_func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'async test_async_func()'
    var_13 = bool('async test_async_func()' in var_0.doc['test_module.test_async_func'])
    assert var_13 is True
    var_14 = '*Full name:* `test_module.test_async_func`'
    var_15 = bool('*Full name:* `test_module.test_async_func`' in var_0.doc['test_module.test_async_func'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'class TestClass'
    var_9 = bool('class TestClass' in var_0.doc['test_module.TestClass'])
    assert var_9 is True
    var_10 = '*Full name:* `test_module.TestClass`'
    var_11 = bool('*Full name:* `test_module.TestClass`' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'decorator'
    var_12 = []
    var_13 = 'Decorators'
    var_14 = bool('Decorators' in var_0.doc['test_module.test_func'])
    assert var_14 is True
    var_15 = '@decorator'
    var_16 = bool('@decorator' in var_0.doc['test_module.test_func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'Test docstring'
    var_11 = []
    var_12 = []
    var_13 = 'Test docstring'
    var_14 = bool('Test docstring' in var_0.docstring['test_module.test_func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'OuterClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'inner_func'
    var_9 = []
    var_10 = []
    var_11 = None
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = 'test_module.OuterClass.inner_func'
    var_19 = bool('test_module.OuterClass.inner_func' in var_0.doc)
    assert var_19 is True
    var_20 = 'inner_func()'
    var_21 = bool('inner_func()' in var_0.doc['test_module.OuterClass.inner_func'])
    assert var_21 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_visit_Constant_returns_node_for_non_string_constant. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_returns_node_for_invalid_string_syntax. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_visits_parsed_expression_for_valid_string. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = [var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax @#$%'
    var_4 = [var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'some_name'
    var_4 = [var_3]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_public_returns_true_for_public_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_returns_false_for_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_returns_true_for_name_in_all. Retrieved 5/7 statements.
# Partially parsed test_is_public_returns_true_for_root_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_returns_false_for_non_public_name_not_in_all. Retrieved 6/8 statements.


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
    var_2 = 'root.name'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

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
    var_2 = set()
    var_3 = 'root.other'
    var_4 = 'root.name'
    var_5 = var_0.is_public(var_4)
    assert var_5 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 10/23 statements.
# Partially parsed test_class_api_with_enums. Retrieved 10/31 statements.
# Partially parsed test_class_api_with_members. Retrieved 11/28 statements.
# Partially parsed test_class_api_with_no_bases_or_members. Retrieved 12/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 0
    var_4 = 'MyClass'
    var_5 = 'BaseClass'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root.MyClass'
    var_10 = []
    var_11 = []
    var_12 = 'root.MyClass'
    var_13 = bool('root.MyClass' in var_0.doc)
    assert var_13 is True
    var_14 = var_0.doc['root.MyClass']
    assert var_14 == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 0
    var_4 = 'MyEnum'
    var_5 = 'Enum'
    var_6 = []
    var_7 = 'A'
    var_8 = []
    var_9 = 'int'
    var_10 = []
    var_11 = []
    var_12 = 'root.MyEnum'
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'root.MyEnum'
    var_17 = bool('root.MyEnum' in var_0.doc)
    assert var_17 is True
    var_18 = var_0.doc['root.MyEnum']
    assert var_18 == '### class MyEnum\n\n*Full name:* `root.MyEnum`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 0
    var_4 = 'MyClass'
    var_5 = []
    var_6 = 'attr'
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = []
    var_11 = 'root.MyClass'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 'root.MyClass'
    var_16 = bool('root.MyClass' in var_0.doc)
    assert var_16 is True
    var_17 = var_0.doc['root.MyClass']
    assert var_17 == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 0
    var_4 = 'MyClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root.MyClass'
    var_10 = []
    var_11 = []
    var_12 = var_0.class_api(var_1, var_9, var_10, var_11)
    var_13 = 'root.MyClass'
    var_14 = bool('root.MyClass' in var_0.doc)
    assert var_14 is True
    var_15 = var_0.doc['root.MyClass']
    assert var_15 == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_visit_Attribute_with_typing_prefix. Retrieved 7/13 statements.
# Partially parsed test_visit_Attribute_without_typing_prefix. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'alias'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'typing'
    var_6 = []
    var_7 = 'List'
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'alias'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'module'
    var_6 = []
    var_7 = 'List'
    var_8 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 13/18 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 13/18 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'b'
    var_5 = [var_4, var_2]
    var_6 = [var_2, var_2]
    var_7 = 'root'
    var_8 = 'name'
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `name`\n\n| a | b | return |\n|:---:|:---:|:---:|\n|  |  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*args'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'root'
    var_5 = 'name'
    var_6 = []
    var_7 = []
    var_8 = 0
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = False
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `name`\n\n| *args | return |\n|:---:|:---:|\n|  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '**kwargs'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'root'
    var_5 = 'name'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 0
    var_11 = []
    var_12 = False
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `name`\n\n| **kwargs | return |\n|:---:|:---:|\n|  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = [var_4, var_2]
    var_6 = 'root'
    var_7 = 'name'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = True
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `name`\n\n| self | a | return |\n|:---:|:---:|:---:|\n| Self |  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = [var_4, var_2]
    var_6 = 'root'
    var_7 = 'name'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = True
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `name`\n\n| cls | a | return |\n|:---:|:---:|:---:|\n| type[Self] |  |  |\n\n'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 9/15 statements.
# Partially parsed test_compile_without_toc. Retrieved 8/14 statements.
# Partially parsed test_compile_with_constants. Retrieved 11/17 statements.
# Partially parsed test_compile_with_non_public_members. Retrieved 12/18 statements.
# Partially parsed test_compile_with_magic_methods. Retrieved 12/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '## Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = set()
    var_6 = 0
    var_7 = var_1.compile()
    var_8 = '**Table of contents:**\n    + [`module`](#module)\n\n## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '## Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = set()
    var_6 = var_1.compile()
    var_7 = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '## Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = set()
    var_6 = 'module.CONST'
    var_7 = 1
    var_8 = 'int'
    var_9 = var_1.compile()
    var_10 = '## Module `module`\n<a id="module"></a>\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n\nModule docstring\n'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module._private'
    var_4 = '## Module `module`\n<a id="module"></a>\n\n'
    var_5 = '### _private()\n\n*Full name:* `module._private`\n<a id="module._private"></a>\n\n'
    var_6 = 'Module docstring'
    var_7 = 'Private function'
    var_8 = set()
    var_9 = 1
    var_10 = var_1.compile()
    var_11 = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = 'module.__magic__'
    var_4 = '## Module `module`\n<a id="module"></a>\n\n'
    var_5 = '### __magic__()\n\n*Full name:* `module.__magic__`\n<a id="module.__magic__"></a>\n\n'
    var_6 = 'Module docstring'
    var_7 = 'Magic method'
    var_8 = set()
    var_9 = 1
    var_10 = var_1.compile()
    var_11 = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_12 = bool(var_10 == var_11)
    assert var_12 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_const_type_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_tuple_empty. Retrieved 2/4 statements.
# Partially parsed test_const_type_tuple_int. Retrieved 3/8 statements.
# Partially parsed test_const_type_tuple_mixed_types. Retrieved 3/8 statements.
# Partially parsed test_const_type_list_empty. Retrieved 2/4 statements.
# Partially parsed test_const_type_list_str. Retrieved 3/8 statements.
# Partially parsed test_const_type_set_empty. Retrieved 1/3 statements.
# Partially parsed test_const_type_set_float. Retrieved 2/7 statements.
# Partially parsed test_const_type_dict_empty. Retrieved 2/4 statements.
# Partially parsed test_const_type_dict_str_int. Retrieved 2/8 statements.
# Partially parsed test_const_type_call_builtin_int. Retrieved 3/8 statements.
# Partially parsed test_const_type_call_builtin_str. Retrieved 3/8 statements.
# Partially parsed test_const_type_call_non_builtin. Retrieved 3/6 statements.
# Partially parsed test_const_type_unsupported_node. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = None

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'hello'
    var_3 = [var_2]
    var_4 = None

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = None

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 1.0
    var_1 = [var_0]
    var_2 = 2.0
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]

def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = '42'
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = 'str'
    var_1 = [var_0]
    var_2 = 42
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = 'custom_func'
    var_1 = [var_0]
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'unsupported'
    var_1 = [var_0]



# Parsed testcases at query #4
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_public_predicate_evaluates_to_true. Retrieved 7/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = 'some_doc'
    var_3 = 'module'
    var_4 = 1
    var_5 = set()
    var_6 = var_0.is_public(var_1)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enums. Retrieved 10/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'object'
    var_6 = []
    var_7 = []
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'object'
    var_11 = bool('object' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = []
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = 'enum.Enum'
    var_11 = bool('enum.Enum' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: x: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = module_1.parse(var_2)
    var_6 = var_5.body[var_4]
    var_7 = 'test_module.A'
    var_8 = []
    var_9 = var_6.body
    var_10 = var_0.class_api(var_1, var_7, var_8, var_9)
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.A'])
    assert var_12 is True
    var_13 = 'x'
    var_14 = bool('x' in var_0.doc['test_module.A'])
    assert var_14 is True
    var_15 = 'int'
    var_16 = bool('int' in var_0.doc['test_module.A'])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A(enum.Enum): X = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = module_1.parse(var_2)
    var_6 = var_5.body[var_4]
    var_7 = 'test_module.A'
    var_8 = 'enum.Enum'
    var_9 = []
    var_10 = var_6.body
    var_11 = 'Enums'
    var_12 = bool('Enums' in var_0.doc['test_module.A'])
    assert var_12 is True
    var_13 = 'X'
    var_14 = bool('X' in var_0.doc['test_module.A'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: _x: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = module_1.parse(var_2)
    var_6 = var_5.body[var_4]
    var_7 = 'test_module.A'
    var_8 = []
    var_9 = var_6.body
    var_10 = var_0.class_api(var_1, var_7, var_8, var_9)
    var_11 = 'Members'
    var_12 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_12 is True
    var_13 = '_x'
    var_14 = bool('_x' not in var_0.doc['test_module.A'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: x: int = 1; del x'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 0
    var_5 = module_1.parse(var_2)
    var_6 = var_5.body[var_4]
    var_7 = 'test_module.A'
    var_8 = []
    var_9 = var_6.body
    var_10 = var_0.class_api(var_1, var_7, var_8, var_9)
    var_11 = 'Members'
    var_12 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_12 is True
    var_13 = 'x'
    var_14 = bool('x' not in var_0.doc['test_module.A'])
    assert var_14 is True



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_name_in_all. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.public_name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root._private_name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.__magic__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.public_name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.public_name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['`1`', '`2`', '`3`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['`1`', ' ', '`3`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a & b'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['<code>a &amp; b</code>'])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a | b'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['<code>a &#124; b</code>'])
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_api_method_creates_function_doc. Retrieved 11/14 statements.
# Partially parsed test_api_method_creates_async_function_doc. Retrieved 11/14 statements.
# Partially parsed test_api_method_creates_class_doc. Retrieved 7/9 statements.
# Partially parsed test_api_method_includes_decorators. Retrieved 11/17 statements.
# Partially parsed test_api_method_includes_bases_for_class. Retrieved 7/12 statements.
# Partially parsed test_api_method_includes_members_for_class. Retrieved 10/18 statements.
# Partially parsed test_api_method_includes_enums_for_enum_class. Retrieved 8/19 statements.
# Partially parsed test_api_method_includes_docstring. Retrieved 11/17 statements.
# Partially parsed test_api_method_handles_nested_classes. Retrieved 11/15 statements.
# Partially parsed test_api_method_handles_nested_functions. Retrieved 18/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = 'root'
    var_12 = 'test_func()'
    var_13 = bool('test_func()' in var_0.doc['root.test_func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_async_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = 'root'
    var_12 = 'async test_async_func()'
    var_13 = bool('async test_async_func()' in var_0.doc['root.test_async_func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'class TestClass'
    var_9 = bool('class TestClass' in var_0.doc['root.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'decorator'
    var_10 = []
    var_11 = None
    var_12 = 'root'
    var_13 = 'Decorators'
    var_14 = bool('Decorators' in var_0.doc['root.test_func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'BaseClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['root.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'member'
    var_2 = 'int'
    var_3 = []
    var_4 = None
    var_5 = 1
    var_6 = 'TestClass'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['root.TestClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'ENUM_VALUE'
    var_2 = 1
    var_3 = []
    var_4 = 'TestEnum'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'Enums'
    var_11 = bool('Enums' in var_0.doc['root.TestEnum'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Test docstring"""'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = None
    var_12 = 'root'
    var_13 = 'Test docstring'
    var_14 = bool('Test docstring' in var_0.docstring['root.test_func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'InnerClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'OuterClass'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'class InnerClass'
    var_13 = bool('class InnerClass' in var_0.doc['root.OuterClass.InnerClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'inner_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = 'outer_func'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = 'root'
    var_20 = 'inner_func()'
    var_21 = bool('inner_func()' in var_0.doc['root.outer_func.inner_func'])
    assert var_21 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 15/42 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_class'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 42
    var_6 = []
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = (var_9, var_10)
    var_12 = []
    var_13 = 'arr'
    var_14 = []
    var_15 = 0
    var_16 = []
    var_17 = 10
    var_18 = []
    var_19 = []
    var_20 = 'x'
    var_21 = bool('x' not in var_0.doc[var_2])
    assert var_21 is True
    var_22 = 'y'
    var_23 = bool('y' not in var_0.doc[var_2])
    assert var_23 is True
    var_24 = 'a'
    var_25 = bool('a' not in var_0.doc[var_2])
    assert var_25 is True
    var_26 = 'b'
    var_27 = bool('b' not in var_0.doc[var_2])
    assert var_27 is True
    var_28 = 'arr'
    var_29 = bool('arr' not in var_0.doc[var_2])
    assert var_29 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_parse_handles_imports_correctly. Retrieved 4/5 statements.
# Partially parsed test_parse_handles_assignments_correctly. Retrieved 4/5 statements.
# Partially parsed test_parse_handles_function_definitions_correctly. Retrieved 4/5 statements.
# Partially parsed test_parse_handles_class_definitions_correctly. Retrieved 4/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.doc['module_name']
    assert var_4 == '# Module `module_name`\n\n'
    var_5 = var_0.level['module_name']
    assert var_5 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.root['module_name']
    assert var_4 == 'module_name'
    var_5 = set()
    var_6 = var_0.imp['module_name']
    var_7 = bool(var_0.imp['module_name'] == var_5)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""This is a docstring."""'
    var_2 = 'module_name'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'This is a docstring.'
    var_5 = bool('This is a docstring.' in var_0.docstring['module_name'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'import os'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'x = 10'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'def func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'class MyClass: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'MyClass'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/10 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 5/14 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 5/13 statements.
# Partially parsed test_globals_with_uppercase_name. Retrieved 4/10 statements.
# Partially parsed test_globals_with___all__. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.x']
    assert var_7 == '42'
    var_8 = var_0.const['root.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'test'
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.y']
    assert var_5 == "'test'"
    var_6 = var_0.const['root.y']
    assert var_6 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'float'
    var_5 = 'root'
    var_6 = var_0.alias['root.z']
    assert var_6 == '3.14'
    var_7 = var_0.const['root.z']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arr'
    var_2 = []
    var_3 = 0
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = 'root.arr[0]'
    var_9 = bool('root.arr[0]' not in var_0.alias)
    assert var_9 is True
    var_10 = 'root.arr[0]'
    var_11 = bool('root.arr[0]' not in var_0.const)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = []
    var_5 = 'root'
    var_6 = 'root.a'
    var_7 = bool('root.a' not in var_0.alias)
    assert var_7 is True
    var_8 = 'root.b'
    var_9 = bool('root.b' not in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 100
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.CONST']
    assert var_5 == '100'
    var_6 = var_0.const['root.CONST']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'func1'
    var_3 = []
    var_4 = 'func2'
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.imp['root']
    var_8 = bool(var_0.imp['root'] == {'root.func1', 'root.func2'})
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 4/10 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 7/11 statements.
# Partially parsed test_visit_Name_with_alias_and_typevar. Retrieved 7/11 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'self_ty'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'typing.TypeVar'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = ''
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'name'
    var_5 = []



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_walk_body_single_node.
# Failed to parse test_walk_body_if_statement.
# Failed to parse test_walk_body_try_statement.
# Failed to parse test_walk_body_nested_control_structures.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_class_api_predicate_evaluates_false. Retrieved 7/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr'
    var_2 = 42
    var_3 = []
    var_4 = 'int'
    var_5 = 'root'
    var_6 = 'name'
    var_7 = []



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a normal docstring.\nWith multiple lines.\nNo doctest here.'
    var_1 = 'This is a normal docstring.\nWith multiple lines.\nNo doctest here.'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')"
    var_1 = "```python\n>>> print('hello')\n```"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n>>> print(x)'
    var_1 = '```python\n>>> x = 1\n>>> print(x)\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Normal text\n>>> doctest line\nMore normal text'
    var_1 = 'Normal text\n```python\n>>> doctest line\n```\nMore normal text'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Text\n>>> block1\nText\n>>> block2\nText'
    var_1 = 'Text\n```python\n>>> block1\n```\nText\n```python\n>>> block2\n```\nText'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Text\n>>> last line'
    var_1 = 'Text\n```python\n>>> last line\n```'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 9/16 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 9/20 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'B'
    var_6 = []
    var_7 = []
    var_8 = 'B'
    var_9 = bool('B' in var_0.doc['test_module.A'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = 'X'
    var_8 = 1
    var_9 = []
    var_10 = 'X'
    var_11 = bool('X' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = 'x'
    var_11 = bool('x' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = 'x'
    var_11 = bool('x' not in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = []
    var_6 = '_x'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = '_x'
    var_11 = bool('_x' not in var_0.doc['test_module.A'])
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_visit_Attribute_with_typing_prefix. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_without_typing_prefix. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_with_non_name_value. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_predicate_at_line_7_evaluates_to_true.




# Parsed testcases at query #21
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is False
    var_5 = var_3.b_level
    assert var_5 == 2
    var_6 = var_3.toc
    assert var_6 is True
    var_7 = var_3.level
    var_8 = bool(var_3.level == {})
    assert var_8 is True
    var_9 = var_3.doc
    var_10 = bool(var_3.doc == {})
    assert var_10 is True
    var_11 = var_3.docstring
    var_12 = bool(var_3.docstring == {})
    assert var_12 is True
    var_13 = var_3.imp
    var_14 = bool(var_3.imp == {})
    assert var_14 is True
    var_15 = var_3.root
    var_16 = bool(var_3.root == {})
    assert var_16 is True
    var_17 = var_3.alias
    var_18 = bool(var_3.alias == {})
    assert var_18 is True
    var_19 = var_3.const
    var_20 = bool(var_3.const == {})
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/17 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 10/21 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 10/17 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 9/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'B'
    var_6 = []
    var_7 = []
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.A'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = []
    var_8 = 'Enums'
    var_9 = bool('Enums' in var_0.doc['test_module.A'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 1
    var_9 = 'test_module.A'
    var_10 = []
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.A'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 1
    var_9 = 'test_module.A'
    var_10 = []
    var_11 = 'Members'
    var_12 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_x'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 1
    var_9 = 'test_module.A'
    var_10 = []
    var_11 = 'Members'
    var_12 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'X'
    var_5 = 1
    var_6 = [var_5]
    var_7 = None
    var_8 = 'test_module.A'
    var_9 = 'enum.Enum'
    var_10 = []
    var_11 = 'Enums'
    var_12 = bool('Enums' in var_0.doc['test_module.A'])
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_api_has_default_true. Retrieved 9/11 statements.
# Partially parsed test_func_api_has_default_false. Retrieved 9/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'name'
    var_9 = False
    var_10 = 'items=[ann]'
    var_11 = bool('items=[ann]' in var_0.doc['name'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'test'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'name'
    var_9 = False
    var_10 = 'items=[ann, _defaults(default)]'
    var_11 = bool('items=[ann, _defaults(default)]' in var_0.doc['name'])
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 6/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 1
    var_8 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_func_ann. Retrieved 12/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'self'
    var_4 = 'Self'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = []
    var_9 = 'y'
    var_10 = 'str'
    var_11 = []
    var_12 = 'return'
    var_13 = 'bool'
    var_14 = []
    var_15 = 'root'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_api_method_with_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_method_with_async_function_def. Retrieved 11/14 statements.
# Partially parsed test_api_method_with_class_def. Retrieved 7/9 statements.
# Partially parsed test_api_method_with_prefix. Retrieved 12/15 statements.
# Partially parsed test_api_method_with_decorators. Retrieved 11/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'test_function()'
    var_13 = bool('test_function()' in var_0.doc['test_module.test_function'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'async test_async_function()'
    var_13 = bool('async test_async_function()' in var_0.doc['test_module.test_async_function'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'class TestClass'
    var_9 = bool('class TestClass' in var_0.doc['test_module.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_method'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'TestClass'
    var_13 = 'test_method()'
    var_14 = bool('test_method()' in var_0.doc['test_module.TestClass.test_method'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_decorated_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'decorator'
    var_12 = []
    var_13 = '@decorator'
    var_14 = bool('@decorator' in var_0.doc['test_module.test_decorated_function'])
    assert var_14 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_has_self_predicate_evaluates_to_true. Retrieved 9/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'test_prefix'
    var_7 = bool(var_6)
    var_8 = '@staticmethod'
    var_9 = '@'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 14/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 9/17 statements.
# Partially parsed test_class_api_with_enums. Retrieved 9/19 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.A'
    var_5 = []
    var_6 = []
    var_7 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_8 = 'Bases'
    var_9 = bool('Bases' not in var_0.doc['test_module.A'])
    assert var_9 is True
    var_10 = module_0.Parser()
    var_11 = 'class B(A): pass'
    var_12 = var_10.parse(var_1, var_11)
    var_13 = 'test_module.B'
    var_14 = 'A'
    var_15 = []
    var_16 = []
    var_17 = 'Bases'
    var_18 = bool('Bases' in var_10.doc['test_module.B'])
    assert var_18 is True
    var_19 = 'A'
    var_20 = bool('A' in var_10.doc['test_module.B'])
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class C: x: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.C'
    var_5 = []
    var_6 = 'x'
    var_7 = 'int'
    var_8 = []
    var_9 = 1
    var_10 = [var_9]
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.C'])
    assert var_12 is True
    var_13 = 'x'
    var_14 = bool('x' in var_0.doc['test_module.C'])
    assert var_14 is True
    var_15 = 'int'
    var_16 = bool('int' in var_0.doc['test_module.C'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class D(enum.Enum): X = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.D'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = 'X'
    var_8 = 1
    var_9 = [var_8]
    var_10 = None
    var_11 = 'Enums'
    var_12 = bool('Enums' in var_0.doc['test_module.D'])
    assert var_12 is True
    var_13 = 'X'
    var_14 = bool('X' in var_0.doc['test_module.D'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class E: _x: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.E'
    var_5 = []
    var_6 = '_x'
    var_7 = 'int'
    var_8 = []
    var_9 = 1
    var_10 = [var_9]
    var_11 = 'Members'
    var_12 = bool('Members' not in var_0.doc['test_module.E'])
    assert var_12 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_public_evaluates_true. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = 'content'
    var_3 = 'module'
    var_4 = set()
    var_5 = var_0.is_public(var_1)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum.Enum'
    var_2 = 'other.Base'
    var_3 = [var_1, var_2]
    var_4 = 'root'
    var_5 = 'name'
    var_6 = []
    var_7 = var_0.class_api(var_4, var_5, var_3, var_6)
    var_8 = 'Enums'
    var_9 = bool('Enums' in var_0.doc['name'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'other.Base'
    var_2 = [var_1]
    var_3 = 'root'
    var_4 = 'name'
    var_5 = []
    var_6 = var_0.class_api(var_3, var_4, var_2, var_5)
    var_7 = 'Enums'
    var_8 = bool('Enums' not in var_0.doc['name'])
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_globals_ann_assign. Retrieved 5/12 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_assign_without_type_comment. Retrieved 4/10 statements.
# Partially parsed test_globals_assign_tuple. Retrieved 5/15 statements.
# Partially parsed test_globals_assign_list. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.x']
    assert var_7 == '42'
    var_8 = var_0.const['root.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'str'
    var_5 = 'root'
    var_6 = var_0.alias['root.y']
    assert var_6 == "'hello'"
    var_7 = var_0.const['root.y']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.z']
    assert var_5 == '3.14'
    var_6 = var_0.const['root.z']
    assert var_6 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'x'
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.imp['root']
    var_9 = bool(var_0.imp['root'] == {'root.x', 'root.y'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'a'
    var_3 = []
    var_4 = 'b'
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.imp['root']
    var_9 = bool(var_0.imp['root'] == {'root.a', 'root.b'})
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 11/19 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_self_param. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_cls_param. Retrieved 10/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arg1'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = 'arg1'
    var_12 = bool('arg1' in var_0.doc['name'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'arg1'
    var_3 = None
    var_4 = []
    var_5 = 'arg2'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'default'
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = bool('arg1' in var_0.doc['name'] and 'arg2' in var_0.doc['name'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'args'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False
    var_12 = '*args'
    var_13 = bool('*args' in var_0.doc['name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'kwarg1'
    var_5 = []
    var_6 = [var_3]
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = 'kwarg1'
    var_12 = bool('kwarg1' in var_0.doc['name'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False
    var_12 = '**kwargs'
    var_13 = bool('**kwargs' in var_0.doc['name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = 'str'
    var_11 = []
    var_12 = False
    var_13 = 'return'
    var_14 = bool('return' in var_0.doc['name'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True
    var_11 = False
    var_12 = 'self'
    var_13 = bool('self' in var_0.doc['name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True
    var_11 = 'cls'
    var_12 = bool('cls' in var_0.doc['name'])
    assert var_12 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_public_returns_true_when_s_is_root_and_all_l_contains_s. Retrieved 4/8 statements.
# Partially parsed test_is_public_returns_true_when_parent_s_in_all_l. Retrieved 6/10 statements.
# Partially parsed test_is_public_returns_true_when_s_in_all_l. Retrieved 5/9 statements.


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
    var_4 = 'child'
    var_5 = var_0.is_public(var_4)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'child'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/10 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 5/11 statements.
# Partially parsed test_globals_with_assign_to_all. Retrieved 5/14 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 5/13 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 5/13 statements.
# Partially parsed test_globals_with_non_constant_value. Retrieved 5/13 statements.
# Partially parsed test_globals_with_uppercase_name. Retrieved 4/10 statements.
# Partially parsed test_globals_with_uppercase_name_and_existing_const. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.alias['root.x']
    assert var_7 == '42'
    var_8 = var_0.const['root.x']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.y']
    assert var_5 == "'hello'"
    var_6 = var_0.const['root.y']
    assert var_6 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 3.14
    var_3 = []
    var_4 = 'float'
    var_5 = 'root'
    var_6 = var_0.alias['root.z']
    assert var_6 == '3.14'
    var_7 = var_0.const['root.z']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'x'
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = 'root'
    var_7 = var_0.imp['root']
    var_8 = bool(var_0.imp['root'] == {'root.x', 'root.y'})
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = []
    var_3 = 0
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = bool(not var_0.alias)
    assert var_8 is True
    var_9 = bool(not var_0.const)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = []
    var_5 = 'root'
    var_6 = bool(not var_0.alias)
    assert var_6 is True
    var_7 = bool(not var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.x']
    assert var_5 == 'y'
    var_6 = 'root.x'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 100
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.CONST']
    assert var_5 == '100'
    var_6 = var_0.const['root.CONST']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONST'
    var_2 = 100
    var_3 = []
    var_4 = 'root'
    var_5 = var_0.alias['root.CONST']
    assert var_5 == '100'
    var_6 = var_0.const['root.CONST']
    assert var_6 == 'float'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_public_returns_false_when_no_children_with_public_family. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'doc'
    var_4 = 'const'
    var_5 = 'root.child'
    var_6 = var_0.is_public(var_1)
    var_7 = bool(not var_6)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_visit_Attribute_with_Name_value. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = []
    var_2 = 'Any'
    var_3 = []
    var_4 = 'root'
    var_5 = {}
    var_6 = module_0.Resolver(var_4, var_5)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_tuple_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_mixed_tuple. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_floats. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_set_of_strings. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_dict_of_int_to_str. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_builtin_func_call. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unknown_node. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1.1
    var_1 = []
    var_2 = 2.2
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'y'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 'a'
    var_5 = []
    var_6 = 'b'
    var_7 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_func_ann_with_self_and_cls_method. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_self_no_cls_method. Retrieved 8/15 statements.
# Partially parsed test_func_ann_without_self. Retrieved 5/11 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 5/9 statements.
# Partially parsed test_func_ann_with_multiple_args. Retrieved 11/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = True
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = 'y'
    var_8 = 'str'
    var_9 = []
    var_10 = 'z'
    var_11 = []
    var_12 = 'root'
    var_13 = True
    var_14 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_const_type_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_tuple. Retrieved 1/3 statements.
# Partially parsed test_const_type_list. Retrieved 1/3 statements.
# Partially parsed test_const_type_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_call_with_valid_func. Retrieved 3/7 statements.
# Partially parsed test_const_type_call_with_invalid_func. Retrieved 3/7 statements.
# Partially parsed test_const_type_constant. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'invalid_func'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_parser_new_constructor. Retrieved 3/4 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.link
    assert var_1 is True
    var_2 = var_0.b_level
    assert var_2 == 1
    var_3 = var_0.toc
    assert var_3 is False
    var_4 = var_0.level
    var_5 = bool(var_0.level == {})
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = bool(var_0.doc == {})
    assert var_7 is True
    var_8 = var_0.docstring
    var_9 = bool(var_0.docstring == {})
    assert var_9 is True
    var_10 = var_0.imp
    var_11 = bool(var_0.imp == {})
    assert var_11 is True
    var_12 = var_0.root
    var_13 = bool(var_0.root == {})
    assert var_13 is True
    var_14 = var_0.alias
    var_15 = bool(var_0.alias == {})
    assert var_15 is True
    var_16 = var_0.const
    var_17 = bool(var_0.const == {})
    assert var_17 is True

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = var_1.toc
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = var_1.toc
    assert var_3 is False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_func_ann_yields_resolved_annotation_when_annotation_exists. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_arg'
    var_2 = 'int'
    var_3 = []
    var_4 = 'root'
    var_5 = False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_visit_Constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_invalid_syntax_string. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_valid_string. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = [var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = [var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'valid_syntax'
    var_4 = [var_3]



# Parsed testcases at query #43
#--------------------------

# Failed to parse test__e_type_empty_sequence.
# Partially parsed test__e_type_empty_element. Retrieved 1/2 statements.
# Partially parsed test__e_type_single_element_single_constant. Retrieved 1/7 statements.
# Partially parsed test__e_type_multiple_elements_single_constant. Retrieved 2/10 statements.
# Partially parsed test__e_type_multiple_elements_multiple_constants. Retrieved 2/9 statements.
# Partially parsed test__e_type_multiple_elements_different_types. Retrieved 2/9 statements.
# Partially parsed test__e_type_non_constant_element. Retrieved 2/6 statements.
# Partially parsed test__e_type_mixed_constants_and_non_constants. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2



