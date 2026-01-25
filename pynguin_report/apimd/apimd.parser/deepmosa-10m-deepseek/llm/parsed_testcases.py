####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'import module'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'root.module': 'module'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'import module as mod'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'root.mod': 'module'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'from package import module'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'root.module': 'package.module'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'from package import module as mod'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'root.mod': 'package.module'})
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.sub'
    var_2 = 'from ..package import module'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.alias
    var_5 = bool(var_0.alias == {'root.sub.module': 'package.module'})
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom sys import path\n\nCONSTANT = 42\n\ndef func():\n    pass\n\nclass Cls:\n    pass\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module'
    var_5 = bool('module' in var_0.doc)
    assert var_5 is True
    var_6 = 'module.func'
    var_7 = bool('module.func' in var_0.doc)
    assert var_7 is True
    var_8 = 'module.Cls'
    var_9 = bool('module.Cls' in var_0.doc)
    assert var_9 is True
    var_10 = 'module.CONSTANT'
    var_11 = bool('module.CONSTANT' in var_0.const)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = ''
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module'
    var_5 = bool('module' in var_0.doc)
    assert var_5 is True
    var_6 = var_0.doc
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_0.const
    var_9 = len(var_8)
    assert var_9 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom sys import path\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module.os'
    var_5 = bool('module.os' in var_0.alias)
    assert var_5 is True
    var_6 = 'module.path'
    var_7 = bool('module.path' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nCONSTANT = 42\nANOTHER_CONSTANT = "value"\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module.CONSTANT'
    var_5 = bool('module.CONSTANT' in var_0.const)
    assert var_5 is True
    var_6 = 'module.ANOTHER_CONSTANT'
    var_7 = bool('module.ANOTHER_CONSTANT' in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\ndef func():\n    pass\n\nasync def async_func():\n    pass\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module.func'
    var_5 = bool('module.func' in var_0.doc)
    assert var_5 is True
    var_6 = 'module.async_func'
    var_7 = bool('module.async_func' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Cls:\n    pass\n\nclass AnotherCls:\n    pass\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module.Cls'
    var_5 = bool('module.Cls' in var_0.doc)
    assert var_5 is True
    var_6 = 'module.AnotherCls'
    var_7 = bool('module.AnotherCls' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\ndef func():\n    """This is a docstring."""\n    pass\n\nclass Cls:\n    """This is a class docstring."""\n    pass\n    '
    var_2 = 'module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'module.func'
    var_5 = bool('module.func' in var_0.docstring)
    assert var_5 is True
    var_6 = 'module.Cls'
    var_7 = bool('module.Cls' in var_0.docstring)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_walk_body_simple_statement.
# Failed to parse test_walk_body_if_statement.
# Failed to parse test_walk_body_try_statement.
# Partially parsed test_walk_body_nested_statements. Retrieved 1/30 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__attr_with_single_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_with_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test__attr_with_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test__attr_with_nonexistent_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test__attr_with_empty_attr_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'x'

def test_case_0():
    var_0 = 2
    var_1 = 'inner.y'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'inner.nonexistent'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'any.attribute'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_imports_predicate_evaluates_to_false. Retrieved 3/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'root'
    var_3 = var_0.alias
    var_4 = bool(var_0.alias == {'root.module_name': 'module_name'})
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_api_method_with_function_def. Retrieved 6/9 statements.
# Partially parsed test_api_method_with_async_function_def. Retrieved 6/9 statements.
# Partially parsed test_api_method_with_class_def. Retrieved 6/8 statements.
# Partially parsed test_api_method_with_decorators. Retrieved 6/11 statements.
# Partially parsed test_api_method_with_prefix. Retrieved 7/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'test_func()'
    var_8 = bool('test_func()' in var_0.doc['test_module.test_func'])
    assert var_8 is True
    var_9 = '*Full name:* `test_module.test_func`'
    var_10 = bool('*Full name:* `test_module.test_func`' in var_0.doc['test_module.test_func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'async test_async_func()'
    var_8 = bool('async test_async_func()' in var_0.doc['test_module.test_async_func'])
    assert var_8 is True
    var_9 = '*Full name:* `test_module.test_async_func`'
    var_10 = bool('*Full name:* `test_module.test_async_func`' in var_0.doc['test_module.test_async_func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'class TestClass'
    var_8 = bool('class TestClass' in var_0.doc['test_module.TestClass'])
    assert var_8 is True
    var_9 = '*Full name:* `test_module.TestClass`'
    var_10 = bool('*Full name:* `test_module.TestClass`' in var_0.doc['test_module.TestClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = 'decorator'
    var_6 = []
    var_7 = None
    var_8 = '@decorator'
    var_9 = bool('@decorator' in var_0.doc['test_module.test_func'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'TestClass'
    var_8 = 'test_module.TestClass.test_func'
    var_9 = bool('test_module.TestClass.test_func' in var_0.doc['test_module.TestClass.test_func'])
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 6/11 statements.
# Partially parsed test_class_api_with_enums. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 7/15 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 7/15 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'Base'
    var_5 = []
    var_6 = []
    var_7 = var_0.doc['root.Class']
    assert var_7 == '| Bases |\n|:---:|\n| `Base` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'enum.Enum'
    var_5 = []
    var_6 = 'ENUM_VALUE'
    var_7 = 1
    var_8 = []
    var_9 = var_0.doc['root.Class']
    assert var_9 == '| Enums |\n|:---:|\n| ENUM_VALUE |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'member'
    var_6 = 'int'
    var_7 = []
    var_8 = var_0.doc['root.Class']
    assert var_8 == '| Members | Type |\n|:---:|:---:|\n| `member` | `int` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = '_private'
    var_6 = 'int'
    var_7 = []
    var_8 = var_0.doc['root.Class']
    assert var_8 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.Class'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'member'
    var_6 = 'int'
    var_7 = []
    var_8 = var_0.doc['root.Class']
    assert var_8 == ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 7/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_name'
    var_3 = 'test_attr'
    var_4 = 42
    var_5 = []
    var_6 = []
    var_7 = 0



# Parsed testcases at query #10
#--------------------------

# Failed to parse test__e_type_empty_input.
# Partially parsed test__e_type_single_empty_element. Retrieved 1/2 statements.
# Partially parsed test__e_type_multiple_empty_elements. Retrieved 2/3 statements.
# Failed to parse test__e_type_single_element_with_non_constant.
# Partially parsed test__e_type_single_element_with_constants_of_same_type. Retrieved 2/9 statements.
# Partially parsed test__e_type_single_element_with_constants_of_different_types. Retrieved 2/9 statements.
# Partially parsed test__e_type_multiple_elements_with_constants_of_same_type. Retrieved 4/14 statements.
# Partially parsed test__e_type_multiple_elements_with_constants_of_different_types. Retrieved 4/14 statements.
# Partially parsed test__e_type_mixed_elements_with_constants_and_non_constants. Retrieved 3/15 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

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
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = 2
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/20 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/20 statements.
# Partially parsed test_globals_with_assign_type_comment. Retrieved 5/17 statements.
# Partially parsed test_globals_with_assign_invalid_targets. Retrieved 5/22 statements.
# Partially parsed test_globals_with_assign_non_name_target. Retrieved 5/16 statements.
# Partially parsed test_globals_with_assign_non_uppercase_target. Retrieved 4/20 statements.
# Partially parsed test_globals_with_assign_all. Retrieved 5/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST'
    var_2 = [var_1]
    var_3 = 'int'
    var_4 = 42
    var_5 = [var_4]
    var_6 = 'root'
    var_7 = var_0.const['root.TEST']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST'
    var_2 = [var_1]
    var_3 = 42
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = var_0.const['root.TEST']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST'
    var_2 = [var_1]
    var_3 = 'dummy'
    var_4 = 'str'
    var_5 = 'root'
    var_6 = var_0.const['root.TEST']
    assert var_6 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST1'
    var_2 = [var_1]
    var_3 = 'TEST2'
    var_4 = [var_3]
    var_5 = 42
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = 'root.TEST1'
    var_9 = bool('root.TEST1' not in var_0.const)
    assert var_9 is True
    var_10 = 'root.TEST2'
    var_11 = bool('root.TEST2' not in var_0.const)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TEST'
    var_2 = 42
    var_3 = [var_2]
    var_4 = [var_1]
    var_5 = 'root'
    var_6 = 'root.TEST'
    var_7 = bool('root.TEST' not in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = [var_1]
    var_3 = 42
    var_4 = [var_3]
    var_5 = 'root'
    var_6 = 'root.test'
    var_7 = bool('root.test' not in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = [var_1]
    var_3 = 'TEST1'
    var_4 = [var_3]
    var_5 = 'TEST2'
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = 'root.TEST1'
    var_9 = bool('root.TEST1' in var_0.imp['root'])
    assert var_9 is True
    var_10 = 'root.TEST2'
    var_11 = bool('root.TEST2' in var_0.imp['root'])
    assert var_11 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_e_type_empty_elements.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_attr_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nested.nonexistent'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 11/19 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwonlyargs_and_kwarg. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_self_and_classmethod. Retrieved 10/14 statements.


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
    var_9 = 'name'
    var_10 = False
    var_11 = var_0.doc['name']
    assert var_11 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| / |\n|:---:|\n|  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'None'
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| x | y |\n|:---:|:---:|\n|  |  |\n|  | None |\n\n'

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
    var_12 = var_0.doc['name']
    assert var_12 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| *args |\n|:---:|\n|  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'x'
    var_5 = []
    var_6 = [var_3]
    var_7 = 'kwargs'
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| * | x | **kwargs |\n|:---:|:---:|:---:|\n|  |  |  |\n|  |  |  |\n\n'

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
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| return |\n|:---:|\n| `int` |\n\n'

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
    var_11 = var_0.doc['name']
    assert var_11 == '### name()\n\n*Full name:* `name`\n<a id="name"></a>\n\n| self |\n|:---:|\n| `type[Self]` |\n\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 9/16 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 9/20 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 9/16 statements.


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
    var_10 = bool('x' in var_0.doc['test_module.A'] and 'int' in var_0.doc['test_module.A'])
    assert var_10 is True

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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_node_posonlyargs_evaluates_to_true. Retrieved 10/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = 'arg1'
    var_3 = module_0.Parser()
    var_4 = 'root'
    var_5 = 'name'
    var_6 = False
    var_7 = var_3.func_api(var_4, var_5, var_1, var_0, has_self=var_6, cls_method=var_6)
    var_8 = var_3.doc[var_5]
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_func_ann_with_self_and_cls_method. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_self_and_no_cls_method. Retrieved 8/15 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 7/13 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 6/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'arg1'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'arg1'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = True
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arg1'
    var_2 = 'int'
    var_3 = []
    var_4 = 'arg2'
    var_5 = 'str'
    var_6 = []
    var_7 = 'root'
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = []
    var_4 = 'arg1'
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'arg1'
    var_2 = None
    var_3 = []
    var_4 = 'arg2'
    var_5 = []
    var_6 = 'root'
    var_7 = False



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_class_api_with_bases_and_enum. Retrieved 10/27 statements.
# Partially parsed test_class_api_without_bases_and_enum. Retrieved 10/24 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/27 statements.
# Partially parsed test_class_api_with_members. Retrieved 12/31 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'name'
    var_4 = 0
    var_5 = 'BaseClass'
    var_6 = []
    var_7 = 'attr1'
    var_8 = 'int'
    var_9 = []
    var_10 = None
    var_11 = 1
    var_12 = 'name'
    var_13 = bool('name' in var_0.doc)
    assert var_13 is True
    var_14 = 'Bases'
    var_15 = bool('Bases' in var_0.doc['name'])
    assert var_15 is True
    var_16 = 'attr1'
    var_17 = bool('attr1' in var_0.doc['name'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'name'
    var_4 = 0
    var_5 = []
    var_6 = 'attr1'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = 1
    var_11 = 'name'
    var_12 = bool('name' in var_0.doc)
    assert var_12 is True
    var_13 = 'Bases'
    var_14 = bool('Bases' not in var_0.doc['name'])
    assert var_14 is True
    var_15 = 'attr1'
    var_16 = bool('attr1' in var_0.doc['name'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'name'
    var_4 = 0
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = 'attr1'
    var_8 = 'int'
    var_9 = []
    var_10 = None
    var_11 = 1
    var_12 = 'name'
    var_13 = bool('name' in var_0.doc)
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc['name'])
    assert var_15 is True
    var_16 = 'attr1'
    var_17 = bool('attr1' in var_0.doc['name'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'name'
    var_4 = 0
    var_5 = []
    var_6 = 'attr1'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = 1
    var_11 = 'attr2'
    var_12 = 42
    var_13 = []
    var_14 = 'name'
    var_15 = bool('name' in var_0.doc)
    assert var_15 is True
    var_16 = 'Members'
    var_17 = bool('Members' in var_0.doc['name'])
    assert var_17 is True
    var_18 = 'attr1'
    var_19 = bool('attr1' in var_0.doc['name'])
    assert var_19 is True
    var_20 = 'attr2'
    var_21 = bool('attr2' in var_0.doc['name'])
    assert var_21 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_instance_of_AnnAssign_and_Name. Retrieved 3/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_api_predicate_evaluates_to_true. Retrieved 12/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = ''
    var_12 = '### test_func()\n\n'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nested.missing_attr'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test__e_type_empty_elements.
# Partially parsed test__e_type_none_element. Retrieved 1/2 statements.
# Partially parsed test__e_type_empty_sequence_element. Retrieved 1/2 statements.
# Failed to parse test__e_type_non_constant_element.
# Partially parsed test__e_type_mixed_type_elements. Retrieved 2/9 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 8/15 statements.
# Partially parsed test_class_api_with_bases_and_enums. Retrieved 7/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.Class'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = var_0.doc['root.Class']
    assert var_6 == '### class Class\n\n*Full name:* `root.Class`\n\n<a id="root.Class"></a>\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = 'Base2'
    var_3 = [var_1, var_2]
    var_4 = 'root'
    var_5 = 'root.Class'
    var_6 = []
    var_7 = var_0.class_api(var_4, var_5, var_3, var_6)
    var_8 = var_0.doc['root.Class']
    assert var_8 == '### class Class\n\n*Full name:* `root.Class`\n\n<a id="root.Class"></a>\n\n| Bases |\n|:---:|\n| Base1 |\n| Base2 |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = 'Base2'
    var_3 = [var_1, var_2]
    var_4 = 'member'
    var_5 = 42
    var_6 = []
    var_7 = 'root'
    var_8 = 'root.Class'
    var_9 = var_0.doc['root.Class']
    assert var_9 == '### class Class\n\n*Full name:* `root.Class`\n\n<a id="root.Class"></a>\n\n| Bases |\n|:---:|\n| Base1 |\n| Base2 |\n\n| Members | Type |\n|:---:|:---:|\n| member | int |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum.Enum'
    var_2 = [var_1]
    var_3 = 'ENUM_VALUE'
    var_4 = 1
    var_5 = []
    var_6 = 'root'
    var_7 = 'root.Class'
    var_8 = var_0.doc['root.Class']
    assert var_8 == '### class Class\n\n*Full name:* `root.Class`\n\n<a id="root.Class"></a>\n\n| Bases |\n|:---:|\n| enum.Enum |\n\n| Enums |\n|:---:|\n| ENUM_VALUE |\n\n'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_evaluates_to_true_for_try_node. Retrieved 4/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_func_api_with_kwarg. Retrieved 13/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = 'kwargs'
    var_9 = []
    var_10 = []
    var_11 = None
    var_12 = False
    var_13 = False
    var_14 = '**kwargs'
    var_15 = bool('**kwargs' in var_0.doc[var_2])
    assert var_15 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_globals_with_multiple_targets. Retrieved 5/11 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 5/11 statements.
# Partially parsed test_globals_with_annassign_and_non_name_target. Retrieved 6/12 statements.
# Partially parsed test_globals_with_assign_and_non_name_target. Retrieved 5/11 statements.
# Partially parsed test_globals_with_assign_and_multiple_targets. Retrieved 5/11 statements.
# Partially parsed test_globals_with_annassign_and_null_value. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'y'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 'root'
    var_8 = var_6.alias
    var_9 = bool(var_6.alias == {})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = []
    var_2 = 'attr'
    var_3 = 42
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.alias
    var_8 = bool(var_5.alias == {})
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = []
    var_2 = 'attr'
    var_3 = 'int'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = module_0.Parser()
    var_8 = 'root'
    var_9 = var_7.alias
    var_10 = bool(var_7.alias == {})
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'obj'
    var_1 = []
    var_2 = 'attr'
    var_3 = 42
    var_4 = []
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.alias
    var_8 = bool(var_5.alias == {})
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'y'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = module_0.Parser()
    var_7 = 'root'
    var_8 = var_6.alias
    var_9 = bool(var_6.alias == {})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'int'
    var_3 = []
    var_4 = None
    var_5 = module_0.Parser()
    var_6 = 'root'
    var_7 = var_5.alias
    var_8 = bool(var_5.alias == {})
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_e_type_empty_sequence. Retrieved 1/2 statements.
# Partially parsed test_e_type_none_elements. Retrieved 2/3 statements.
# Partially parsed test_e_type_single_constant. Retrieved 1/7 statements.
# Partially parsed test_e_type_multiple_constants_same_type. Retrieved 2/9 statements.
# Partially parsed test_e_type_multiple_constants_different_types. Retrieved 2/9 statements.
# Partially parsed test_e_type_nested_sequence_with_constants. Retrieved 2/10 statements.
# Partially parsed test_e_type_nested_sequence_with_non_constants. Retrieved 2/9 statements.
# Partially parsed test_e_type_multiple_sequences. Retrieved 2/10 statements.
# Partially parsed test_e_type_multiple_sequences_different_types. Retrieved 2/10 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]

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
    var_2 = 2
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'not a constant'

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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_api_predicate_evaluates_to_true_for_functiondef. Retrieved 5/8 statements.
# Partially parsed test_api_predicate_evaluates_to_true_for_asyncfunctiondef. Retrieved 5/8 statements.
# Partially parsed test_api_predicate_evaluates_to_true_for_classdef. Retrieved 6/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_tuple_of_constants. Retrieved 3/9 statements.
# Partially parsed test_const_type_with_list_of_constants. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_set_of_constants. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_dict_of_constants. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_call_to_builtin_function. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_call_to_non_builtin_function. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_non_constant_node. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 3.14
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []

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
    var_2 = 42
    var_3 = []

def test_case_0():
    var_0 = 'custom_func'
    var_1 = []
    var_2 = 42
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_visit_Subscript_handles_typing_Union. Retrieved 6/25 statements.
# Partially parsed test_visit_Subscript_handles_typing_Optional. Retrieved 5/19 statements.
# Partially parsed test_visit_Subscript_handles_PEP585_deprecated_names. Retrieved 5/15 statements.
# Partially parsed test_visit_Subscript_returns_node_for_non_typing_Union_or_Optional. Retrieved 6/18 statements.
# Partially parsed test_visit_Subscript_returns_node_for_non_Name_value. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Union'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 'str'
    var_8 = []
    var_9 = []
    var_10 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Optional'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'List'
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Dict'
    var_4 = []
    var_5 = 'str'
    var_6 = []
    var_7 = 'int'
    var_8 = []
    var_9 = []
    var_10 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []
    var_7 = 'int'
    var_8 = []
    var_9 = []



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_func_api_has_default_false. Retrieved 10/16 statements.


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
    var_10 = 'name'
    var_11 = False
    var_12 = 'x'
    var_13 = bool('x' in var_0.doc['name'])
    assert var_13 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_func_api_without_self_and_class_method. Retrieved 8/13 statements.
# Partially parsed test_func_api_with_self_and_class_method. Retrieved 8/13 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 8/13 statements.
# Partially parsed test_func_api_with_posonlyargs_and_kwonlyargs. Retrieved 11/19 statements.
# Partially parsed test_func_api_with_default_values. Retrieved 8/16 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'arg1'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'arg2'
    var_6 = [var_5, var_3]
    var_7 = [var_3, var_3]
    var_8 = 'root'
    var_9 = 'name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'arg1'
    var_6 = [var_5, var_3]
    var_7 = [var_3, var_3]
    var_8 = 'root'
    var_9 = 'name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = '*args'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = '**kwargs'
    var_6 = [var_5, var_3]
    var_7 = [var_3, var_3]
    var_8 = 'root'
    var_9 = 'name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'arg1'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = '/'
    var_6 = [var_5, var_3]
    var_7 = 'arg2'
    var_8 = [var_7, var_3]
    var_9 = '*'
    var_10 = [var_9, var_3]
    var_11 = 'arg3'
    var_12 = [var_11, var_3]
    var_13 = [var_3, var_3, var_3, var_3, var_3]
    var_14 = 'root'
    var_15 = 'name'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'arg1'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'arg2'
    var_6 = [var_5, var_3]
    var_7 = []
    var_8 = 2
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_globals_predicate_evaluates_to_false_when_not_all_or_not_tuple_list. Retrieved 8/29 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'not_all'
    var_3 = []
    var_4 = []
    var_5 = '__all__'
    var_6 = bool('__all__' not in var_0.alias)
    assert var_6 is True
    var_7 = '__all__'
    var_8 = 'x'
    var_9 = []
    var_10 = bool(not var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'other'
    var_12 = 'y'
    var_13 = []
    var_14 = bool('__all__' not in var_0.alias and (not var_0.imp[var_1]))
    assert var_14 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_func_ann_with_annotation_and_not_self_or_star. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_arg'
    var_2 = 'int'
    var_3 = []
    var_4 = 'root'
    var_5 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mod'
    var_2 = var_0.docstring['mod.func']
    assert var_2 == '```python\nDocstring for func\n```'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_parser_new_method_with_parameters. Retrieved 3/4 statements.


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
    var_3 = var_1.b_level
    assert var_3 == 1
    var_4 = var_1.toc
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = var_1.b_level
    assert var_3 == 1
    var_4 = var_1.toc
    assert var_4 is False



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_e_type_with_non_constant_elements.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test__e_type_with_constant_elements. Retrieved 3/14 statements.
# Failed to parse test__e_type_with_empty_elements.
# Partially parsed test__e_type_with_none_element. Retrieved 2/3 statements.
# Partially parsed test__e_type_with_mixed_constant_types. Retrieved 2/12 statements.
# Partially parsed test__e_type_with_non_constant_elements. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
    var_5 = [var_3]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_compile_with_toc. Retrieved 9/15 statements.
# Partially parsed test_compile_without_toc. Retrieved 8/14 statements.
# Partially parsed test_compile_with_constants. Retrieved 10/16 statements.
# Partially parsed test_compile_with_missing_docstring. Retrieved 7/13 statements.
# Partially parsed test_compile_with_multiple_modules. Retrieved 13/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = 0
    var_6 = set()
    var_7 = '**Table of contents:**\n+ [module](#module)\n\n# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_8 = var_1.compile()
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = set()
    var_6 = '# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_7 = var_1.compile()
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n<a id="module"></a>\n\n'
    var_4 = 'Module docstring'
    var_5 = set()
    var_6 = 'module.CONST'
    var_7 = 'int'
    var_8 = '# Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    var_9 = var_1.compile()
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n<a id="module"></a>\n\n'
    var_4 = set()
    var_5 = '# Module `module`\n<a id="module"></a>\n\n'
    var_6 = var_1.compile()
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = '# Module `module1`\n<a id="module1"></a>\n\n'
    var_5 = '# Module `module2`\n<a id="module2"></a>\n\n'
    var_6 = 'Module1 docstring'
    var_7 = 'Module2 docstring'
    var_8 = 0
    var_9 = set()
    var_10 = set()
    var_11 = '**Table of contents:**\n+ [module1](#module1)\n+ [module2](#module2)\n\n# Module `module1`\n<a id="module1"></a>\n\nModule1 docstring\n\n# Module `module2`\n<a id="module2"></a>\n\nModule2 docstring\n'
    var_12 = var_1.compile()
    var_13 = bool(var_12 == var_11)
    assert var_13 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_const_type_call_with_name_or_attribute_func. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = 'builtins'
    var_3 = []
    var_4 = 'str'
    var_5 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_handles_ann_assign_with_name_target_and_value. Retrieved 5/10 statements.
# Partially parsed test_globals_handles_assign_with_name_target_and_value. Retrieved 4/9 statements.
# Partially parsed test_globals_handles_assign_with_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_handles_uppercase_name_as_constant. Retrieved 4/9 statements.
# Partially parsed test_globals_handles___all__assignment_with_tuple. Retrieved 5/13 statements.
# Partially parsed test_globals_handles___all__assignment_with_list. Retrieved 5/13 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 3/8 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 5/11 statements.
# Partially parsed test_globals_ignores_non_tuple_list___all__. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 1
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.alias['root.x']
    assert var_8 == '1'
    var_9 = var_0.const['root.x']
    assert var_9 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 'test'
    var_4 = []
    var_5 = 'root'
    var_6 = var_0.alias['root.y']
    assert var_6 == "'test'"
    var_7 = var_0.const['root.y']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = []
    var_3 = 3.14
    var_4 = []
    var_5 = 'float'
    var_6 = 'root'
    var_7 = var_0.alias['root.z']
    assert var_7 == '3.14'
    var_8 = var_0.const['root.z']
    assert var_8 == 'float'

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'a'
    var_4 = []
    var_5 = 'b'
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.imp['root']
    var_9 = bool(var_0.imp['root'] == {'root.a', 'root.b'})
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
    var_7 = bool(not var_0.imp['root'])
    assert var_7 is True

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
    var_10 = bool(not var_0.imp['root'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'invalid'
    var_4 = []
    var_5 = 'root'
    var_6 = bool(not var_0.imp['root'])
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_Attribute_simple_attribute. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_non_typing_attribute. Retrieved 5/13 statements.
# Partially parsed test_visit_Attribute_nested_attribute. Retrieved 6/18 statements.


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
    var_3 = 'other_module'
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
    var_5 = 'Optional'
    var_6 = []
    var_7 = 'List'
    var_8 = []



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_compile_single_module. Retrieved 2/6 statements.
# Partially parsed test_compile_with_toc. Retrieved 3/7 statements.
# Partially parsed test_compile_with_constants. Retrieved 2/8 statements.
# Partially parsed test_compile_with_private_members. Retrieved 2/6 statements.
# Partially parsed test_compile_with_magic_methods. Retrieved 2/6 statements.
# Partially parsed test_compile_with_nested_modules. Retrieved 2/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '# Module `module`\n\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.compile()
    assert var_2 == '**Table of contents:**\n+ [`module`](#module)\n\n# Module `module`\n\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '# Module `module`\n\n| Constants | Type |\n|:----------|:-----|\n| `CONST` | `int` |\n\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '# Module `module`\n\n\n## sub()\n\n\n'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 4/10 statements.
# Partially parsed test_const_type_with_list_of_mixed_types. Retrieved 4/10 statements.
# Partially parsed test_const_type_with_empty_tuple. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_tuple_of_strs. Retrieved 4/10 statements.
# Partially parsed test_const_type_with_empty_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_set_of_floats. Retrieved 3/9 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_dict_of_int_to_str. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_dict_of_mixed_types. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_builtin_int_call. Retrieved 4/9 statements.
# Partially parsed test_const_type_with_builtin_str_call. Retrieved 4/9 statements.
# Partially parsed test_const_type_with_unknown_call. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = 3.14
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
    var_4 = 3
    var_5 = [var_4]
    var_6 = None

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'two'
    var_3 = [var_2]
    var_4 = 3.0
    var_5 = [var_4]
    var_6 = None

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 'c'
    var_5 = [var_4]
    var_6 = None

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 1.1
    var_1 = [var_0]
    var_2 = 2.2
    var_3 = [var_2]
    var_4 = 3.3
    var_5 = [var_4]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'two'
    var_3 = [var_2]
    var_4 = 3.0
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_6]

def test_case_0():
    var_0 = 'int'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = '42'
    var_4 = [var_3]
    var_5 = []

def test_case_0():
    var_0 = 'str'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = 42
    var_4 = [var_3]
    var_5 = []

def test_case_0():
    var_0 = 'unknown'
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_table_with_single_item. Retrieved 7/8 statements.
# Partially parsed test_table_with_multiple_items. Retrieved 10/11 statements.
# Partially parsed test_table_with_single_column. Retrieved 7/8 statements.
# Partially parsed test_table_with_short_values. Retrieved 10/11 statements.
# Partially parsed test_table_with_mixed_length_values. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 'Header1'
    var_1 = 'Header2'
    var_2 = 'Item1'
    var_3 = 'Item2'
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = [var_0, var_1]
    var_7 = '| Header1 | Header2 |\n|:-------:|:-------:|\n| Item1 | Item2 |\n\n'

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = '1'
    var_3 = '2'
    var_4 = [var_2, var_3]
    var_5 = '3'
    var_6 = '4'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1]
    var_10 = '| A | B |\n|:---:|:---:|\n| 1 | 2 |\n| 3 | 4 |\n\n'

def test_case_0():
    var_0 = 'Column1'
    var_1 = 'Value1'
    var_2 = [var_1]
    var_3 = 'Value2'
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = [var_0]
    var_7 = '| Column1 |\n|:-------:|\n| Value1 |\n| Value2 |\n\n'

def test_case_0():
    var_0 = 'X'
    var_1 = 'Y'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1]
    var_10 = '| X | Y |\n|:---:|:---:|\n| a | b |\n| c | d |\n\n'

def test_case_0():
    var_0 = 'LongHeader'
    var_1 = 'Short'
    var_2 = 'Value'
    var_3 = 'X'
    var_4 = [var_2, var_3]
    var_5 = 'AnotherValue'
    var_6 = 'Y'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1]
    var_10 = '| LongHeader | Short |\n|:----------:|:-----:|\n| Value | X |\n| AnotherValue | Y |\n\n'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 10/16 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 10/14 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/14 statements.


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
    var_9 = 'name'
    var_10 = False
    var_11 = '| x | return |'
    var_12 = bool('| x | return |' in var_0.doc['name'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 1
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = '| x | y | return |'
    var_15 = bool('| x | y | return |' in var_0.doc['name'])
    assert var_15 is True
    var_16 = '|   | 1 |   |'
    var_17 = bool('|   | 1 |   |' in var_0.doc['name'])
    assert var_17 is True

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
    var_12 = '| *args | return |'
    var_13 = bool('| *args | return |' in var_0.doc['name'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'x'
    var_5 = []
    var_6 = 1
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False
    var_12 = '| * | x | return |'
    var_13 = bool('| * | x | return |' in var_0.doc['name'])
    assert var_13 is True
    var_14 = '|   | 1 |   |'
    var_15 = bool('|   | 1 |   |' in var_0.doc['name'])
    assert var_15 is True

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
    var_12 = '| **kwargs | return |'
    var_13 = bool('| **kwargs | return |' in var_0.doc['name'])
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
    var_9 = 'name'
    var_10 = True
    var_11 = False
    var_12 = '| Self | return |'
    var_13 = bool('| Self | return |' in var_0.doc['name'])
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
    var_11 = '| type[Self] | return |'
    var_12 = bool('| type[Self] | return |' in var_0.doc['name'])
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
    var_9 = 'name'
    var_10 = 'int'
    var_11 = []
    var_12 = False
    var_13 = '| return |'
    var_14 = bool('| return |' in var_0.doc['name'])
    assert var_14 is True
    var_15 = '| int |'
    var_16 = bool('| int |' in var_0.doc['name'])
    assert var_16 is True



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello, world!')\nHello, world!"
    var_1 = "```python\n>>> print('Hello, world!')\n```\nHello, world!"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> a = 1\n>>> b = 2\n>>> print(a + b)\n3'
    var_1 = '```python\n>>> a = 1\n>>> b = 2\n>>> print(a + b)\n```\n3'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello')\nHello\nThis is a test."
    var_1 = "```python\n>>> print('Hello')\n```\nHello\nThis is a test."
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "This is a test.\n>>> print('Hello')\nHello\nThis is another test."
    var_1 = "This is a test.\n```python\n>>> print('Hello')\n```\nHello\nThis is another test."
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a test.\nThis is another test.'
    var_1 = 'This is a test.\nThis is another test.'
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('Hello')"
    var_1 = "```python\n>>> print('Hello')\n```"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_api_method_function_def. Retrieved 6/9 statements.
# Partially parsed test_api_method_async_function_def. Retrieved 6/9 statements.
# Partially parsed test_api_method_class_def. Retrieved 6/8 statements.
# Partially parsed test_api_method_with_prefix. Retrieved 7/10 statements.
# Partially parsed test_api_method_with_decorators. Retrieved 7/10 statements.
# Partially parsed test_api_method_with_docstring. Retrieved 7/13 statements.
# Partially parsed test_api_method_with_nested_class. Retrieved 10/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'test_function()'
    var_8 = bool('test_function()' in var_0.doc['root.test_function'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_async_function'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'async test_async_function()'
    var_8 = bool('async test_async_function()' in var_0.doc['root.test_async_function'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'root'
    var_7 = 'class TestClass'
    var_8 = bool('class TestClass' in var_0.doc['root.TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'prefix'
    var_8 = 'prefix.test_function()'
    var_9 = bool('prefix.test_function()' in var_0.doc['root.prefix.test_function'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = []
    var_3 = []
    var_4 = '@decorator'
    var_5 = [var_4]
    var_6 = None
    var_7 = 'root'
    var_8 = '@decorator'
    var_9 = bool('@decorator' in var_0.doc['root.test_function'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_function'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'Docstring'
    var_7 = []
    var_8 = 'root'
    var_9 = 'Docstring'
    var_10 = bool('Docstring' in var_0.docstring['root.test_function'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'test_function'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = 'root'
    var_12 = 'test_function()'
    var_13 = bool('test_function()' in var_0.doc['root.TestClass.test_function'])
    assert var_13 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test__e_type_with_no_elements.
# Partially parsed test__e_type_with_empty_elements. Retrieved 1/2 statements.
# Failed to parse test__e_type_with_non_constant_elements.
# Partially parsed test__e_type_with_single_constant_element. Retrieved 1/7 statements.
# Partially parsed test__e_type_with_multiple_constant_elements_same_type. Retrieved 2/9 statements.
# Partially parsed test__e_type_with_multiple_constant_elements_different_types. Retrieved 2/9 statements.
# Partially parsed test__e_type_with_nested_constant_elements_same_type. Retrieved 4/14 statements.
# Partially parsed test__e_type_with_nested_constant_elements_different_types. Retrieved 4/14 statements.


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
    var_2 = 'a'
    var_3 = [var_2]

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
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 3.0
    var_5 = [var_4]
    var_6 = 4
    var_7 = [var_6]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_const_type_with_constant. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_tuple. Retrieved 3/9 statements.
# Partially parsed test_const_type_with_list. Retrieved 3/9 statements.
# Partially parsed test_const_type_with_set. Retrieved 3/9 statements.
# Partially parsed test_const_type_with_dict. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_call. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_invalid_call. Retrieved 2/5 statements.
# Partially parsed test_const_type_with_non_constant. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_tuple. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_set. Retrieved 1/3 statements.
# Partially parsed test_const_type_with_empty_dict. Retrieved 2/4 statements.
# Partially parsed test_const_type_with_mixed_tuple. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_mixed_list. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_mixed_set. Retrieved 2/7 statements.
# Partially parsed test_const_type_with_mixed_dict. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = 'c'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1.0
    var_1 = [var_0]
    var_2 = 2.0
    var_3 = [var_2]
    var_4 = 3.0
    var_5 = [var_4]

def test_case_0():
    var_0 = 'key1'
    var_1 = [var_0]
    var_2 = 'key2'
    var_3 = [var_2]
    var_4 = 1
    var_5 = [var_4]
    var_6 = 2
    var_7 = [var_6]

def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = '42'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'unknown_func'
    var_1 = [var_0]
    var_2 = []

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = 2
    var_7 = [var_6]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = ">>> print('hello')"
    var_1 = 0
    var_2 = '>>> '
    var_3 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 6/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/16 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_non_public_member. Retrieved 8/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'Base'
    var_5 = []
    var_6 = []
    var_7 = 'Bases'
    var_8 = bool('Bases' in var_0.doc['root.name'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = ''
    var_3 = 'root'
    var_4 = 'enum.Enum'
    var_5 = []
    var_6 = 'ATTR'
    var_7 = 1
    var_8 = []
    var_9 = 'Enums'
    var_10 = bool('Enums' in var_0.doc['root.name'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'attr'
    var_6 = 'int'
    var_7 = []
    var_8 = None
    var_9 = 'Members'
    var_10 = bool('Members' in var_0.doc['root.name'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = 'attr'
    var_6 = 'int'
    var_7 = []
    var_8 = None
    var_9 = 'Members'
    var_10 = bool('Members' not in var_0.doc['root.name'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.name'
    var_2 = ''
    var_3 = 'root'
    var_4 = []
    var_5 = '_attr'
    var_6 = 'int'
    var_7 = []
    var_8 = None
    var_9 = 'Members'
    var_10 = bool('Members' not in var_0.doc['root.name'])
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'import os\nimport sys\n'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = set()
    var_5 = var_0.imp['root']
    var_6 = bool(var_0.imp['root'] == var_4)
    assert var_6 is True
    var_7 = var_0.alias
    var_8 = bool(var_0.alias == {'root.os': 'os', 'root.sys': 'sys'})
    assert var_8 is True
    var_9 = 'x = 10\ny = 20'
    var_10 = var_0.parse(var_1, var_9)
    var_11 = var_0.alias
    var_12 = bool(var_0.alias == {'root.os': 'os', 'root.sys': 'sys', 'root.x': '10', 'root.y': '20'})
    assert var_12 is True
    var_13 = 'class MyClass:\n    pass'
    var_14 = var_0.parse(var_1, var_13)
    var_15 = 'root.MyClass'
    var_16 = bool('root.MyClass' in var_0.doc)
    assert var_16 is True
    var_17 = 'def my_func():\n    pass'
    var_18 = var_0.parse(var_1, var_17)
    var_19 = 'root.my_func'
    var_20 = bool('root.my_func' in var_0.doc)
    assert var_20 is True
    var_21 = 'async def my_async_func():\n    pass'
    var_22 = var_0.parse(var_1, var_21)
    var_23 = 'root.my_async_func'
    var_24 = bool('root.my_async_func' in var_0.doc)
    assert var_24 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 2/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr'
    var_2 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 3/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = []
    var_3 = 0



# Parsed testcases at query #16
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
    var_7 = '__init__'
    var_8 = bool('__init__' not in var_6)
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_walk_body_with_single_node.
# Failed to parse test_walk_body_with_if_node.
# Failed to parse test_walk_body_with_try_node.
# Failed to parse test_walk_body_with_nested_nodes.




# Parsed testcases at query #18
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.doc
    var_2 = var_0.level
    var_3 = var_0.imp
    var_4 = var_0.root
    var_5 = 'example_module'
    var_6 = '\nimport os\nfrom sys import path\ndef example_function():\n    pass\nclass ExampleClass:\n    pass\n'
    var_7 = var_0.parse(var_5, var_6)
    var_8 = var_1['example_module']
    assert var_8 == '## Module `example_module`\n\n'
    var_9 = var_2['example_module']
    assert var_9 == 0
    var_10 = set()
    var_11 = var_3['example_module']
    var_12 = bool(var_3['example_module'] == var_10)
    assert var_12 is True
    var_13 = var_4['example_module']
    assert var_13 == 'example_module'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_public_with_docstring. Retrieved 8/13 statements.
# Partially parsed test_is_public_without_docstring. Retrieved 7/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = ''
    var_3 = 'docstring'
    var_4 = 'module'
    var_5 = 1
    var_6 = set()
    var_7 = var_0.is_public(var_1)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.name'
    var_2 = ''
    var_3 = 'module'
    var_4 = 1
    var_5 = set()
    var_6 = var_0.is_public(var_1)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test__e_type_returns_empty_string_when_elements_empty.
# Partially parsed test__e_type_returns_empty_string_when_element_is_none. Retrieved 1/2 statements.
# Failed to parse test__e_type_returns_empty_string_when_element_contains_non_constant.
# Partially parsed test__e_type_returns_type_name_when_single_element_single_constant. Retrieved 1/7 statements.
# Partially parsed test__e_type_returns_any_when_multiple_constants_with_different_types. Retrieved 2/9 statements.
# Partially parsed test__e_type_returns_type_name_when_multiple_constants_with_same_type. Retrieved 2/9 statements.
# Partially parsed test__e_type_returns_multiple_types_when_multiple_elements. Retrieved 2/10 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_globals_with_assign_node_with_type_comment. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 42
    var_3 = []
    var_4 = 'int'
    var_5 = 'root'
    var_6 = var_0.alias['root.x']
    assert var_6 == '42'
    var_7 = var_0.const
    var_8 = bool(var_0.const == {})
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_walk_body_yields_non_control_flow_nodes. Retrieved 9/35 statements.


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
    var_9 = 'print'
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_self_param. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_cls_param. Retrieved 10/14 statements.


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
    var_11 = 'name'
    var_12 = False
    var_13 = '| x | / |'
    var_14 = bool('| x | / |' in var_0.doc['name'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 1
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = '| x | y |'
    var_15 = bool('| x | y |' in var_0.doc['name'])
    assert var_15 is True
    var_16 = '|   | 1 |'
    var_17 = bool('|   | 1 |' in var_0.doc['name'])
    assert var_17 is True

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
    var_12 = '| *args |'
    var_13 = bool('| *args |' in var_0.doc['name'])
    assert var_13 is True

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
    var_8 = 1
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = '| * | x | y |'
    var_15 = bool('| * | x | y |' in var_0.doc['name'])
    assert var_15 is True
    var_16 = '|   |   | 1 |'
    var_17 = bool('|   |   | 1 |' in var_0.doc['name'])
    assert var_17 is True

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
    var_12 = '| **kwargs |'
    var_13 = bool('| **kwargs |' in var_0.doc['name'])
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
    var_10 = 'int'
    var_11 = []
    var_12 = False
    var_13 = '| return |'
    var_14 = bool('| return |' in var_0.doc['name'])
    assert var_14 is True
    var_15 = '| int |'
    var_16 = bool('| int |' in var_0.doc['name'])
    assert var_16 is True

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
    var_12 = '| Self |'
    var_13 = bool('| Self |' in var_0.doc['name'])
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
    var_11 = '| type[Self] |'
    var_12 = bool('| type[Self] |' in var_0.doc['name'])
    assert var_12 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_members. Retrieved 9/16 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 9/20 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 9/16 statements.


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
    var_10 = 'Enums'
    var_11 = bool('Enums' in var_0.doc['test_module.A'])
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
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc['test_module.A'])
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
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc['test_module.A'])
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
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc['test_module.A'])
    assert var_11 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_globals_node_type_comment_is_not_none. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'x'
    var_2 = 42
    var_3 = []
    var_4 = 'int'
    var_5 = module_0.Parser()
    var_6 = var_5.alias['module.x']
    assert var_6 == '42'
    var_7 = 'module.x'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_const_type_with_call_to_builtin_func. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_builtin_attribute. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'builtins'
    var_1 = []
    var_2 = 'int'
    var_3 = []
    var_4 = []
    var_5 = []



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_empty_element_in_elements. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_api_method_handles_class_def_with_nested_functions. Retrieved 13/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'test_func'
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'test_func'
    var_15 = bool('test_func' in var_0.doc)
    assert var_15 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 12/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'prefix'
    var_13 = var_0.doc['root.prefix.test']
    assert var_13 == '### test()\n\n*Full name:* `root.prefix.test`\n\n'



# Parsed testcases at query #30
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'stmt2'
    var_2 = 'stmt3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.walk_body(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['stmt1', 'stmt2', 'stmt3'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'type'
    var_2 = 'body'
    var_3 = 'orelse'
    var_4 = 'If'
    var_5 = 'stmt2'
    var_6 = [var_5]
    var_7 = 'stmt3'
    var_8 = [var_7]
    var_9 = {var_1: var_4, var_2: var_6, var_3: var_8}
    var_10 = 'stmt4'
    var_11 = [var_0, var_9, var_10]
    var_12 = module_0.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == ['stmt1', 'stmt2', 'stmt3', 'stmt4'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'type'
    var_2 = 'body'
    var_3 = 'handlers'
    var_4 = 'orelse'
    var_5 = 'finalbody'
    var_6 = 'Try'
    var_7 = 'stmt2'
    var_8 = [var_7]
    var_9 = 'stmt3'
    var_10 = [var_9]
    var_11 = {var_2: var_10}
    var_12 = [var_11]
    var_13 = 'stmt4'
    var_14 = [var_13]
    var_15 = 'stmt5'
    var_16 = [var_15]
    var_17 = {var_1: var_6, var_2: var_8, var_3: var_12, var_4: var_14, var_5: var_16}
    var_18 = 'stmt6'
    var_19 = [var_0, var_17, var_18]
    var_20 = module_0.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = bool(var_21 == ['stmt1', 'stmt2', 'stmt3', 'stmt4', 'stmt5', 'stmt6'])
    assert var_22 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'stmt1'
    var_1 = 'type'
    var_2 = 'body'
    var_3 = 'orelse'
    var_4 = 'If'
    var_5 = 'handlers'
    var_6 = 'finalbody'
    var_7 = 'Try'
    var_8 = 'stmt2'
    var_9 = [var_8]
    var_10 = 'stmt3'
    var_11 = [var_10]
    var_12 = {var_2: var_11}
    var_13 = [var_12]
    var_14 = 'stmt4'
    var_15 = [var_14]
    var_16 = 'stmt5'
    var_17 = [var_16]
    var_18 = {var_1: var_7, var_2: var_9, var_5: var_13, var_3: var_15, var_6: var_17}
    var_19 = [var_18]
    var_20 = 'stmt6'
    var_21 = [var_20]
    var_22 = {var_1: var_4, var_2: var_19, var_3: var_21}
    var_23 = 'stmt7'
    var_24 = [var_0, var_22, var_23]
    var_25 = module_0.walk_body(var_24)
    var_26 = list(var_25)
    var_27 = bool(var_26 == ['stmt1', 'stmt2', 'stmt3', 'stmt4', 'stmt5', 'stmt6', 'stmt7'])
    assert var_27 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '"""Module docstring"""\n'
    var_2 = 'root'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'Module docstring'
    var_5 = module_0.doctest(var_4)
    var_6 = var_0.docstring['root']
    var_7 = bool(var_0.docstring['root'] == var_5)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_decorators_list_not_empty. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_root'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = 'decorator'
    var_6 = []
    var_7 = 'Decorators'
    var_8 = bool('Decorators' in var_0.doc[f'{var_1}.test_func'])
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_func_ann_with_self_and_cls_method. Retrieved 5/11 statements.
# Partially parsed test_func_ann_with_self_no_cls_method. Retrieved 6/12 statements.
# Partially parsed test_func_ann_with_annotation. Retrieved 5/11 statements.
# Partially parsed test_func_ann_without_annotation. Retrieved 5/9 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 5/9 statements.
# Partially parsed test_func_ann_multiple_args. Retrieved 13/27 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'root'
    var_5 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'root'
    var_5 = True
    var_6 = False

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
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = []
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'x'
    var_5 = 'int'
    var_6 = []
    var_7 = 'y'
    var_8 = None
    var_9 = []
    var_10 = '*'
    var_11 = []
    var_12 = 'z'
    var_13 = 'str'
    var_14 = []
    var_15 = 'root'
    var_16 = True
    var_17 = False



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'def example_function(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module.example_function'
    var_5 = bool('module.example_function' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'async def example_async_function(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module.example_async_function'
    var_5 = bool('module.example_async_function' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'class ExampleClass: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module.ExampleClass'
    var_5 = bool('module.ExampleClass' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = '@decorator\ndef example_function(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '@decorator'
    var_5 = bool('@decorator' in var_0.doc['module.example_function'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'def example_function(param1, param2): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'param1'
    var_5 = bool('param1' in var_0.doc['module.example_function'])
    assert var_5 is True
    var_6 = 'param2'
    var_7 = bool('param2' in var_0.doc['module.example_function'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'def example_function() -> str: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'return'
    var_5 = bool('return' in var_0.doc['module.example_function'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'class ExampleClass(BaseClass): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'BaseClass'
    var_5 = bool('BaseClass' in var_0.doc['module.ExampleClass'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'class ExampleClass:\n    member: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'member'
    var_5 = bool('member' in var_0.doc['module.ExampleClass'])
    assert var_5 is True
    var_6 = 'int'
    var_7 = bool('int' in var_0.doc['module.ExampleClass'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'class OuterClass:\n    class InnerClass: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module.OuterClass.InnerClass'
    var_5 = bool('module.OuterClass.InnerClass' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'def outer_function():\n    def inner_function(): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'module.outer_function.inner_function'
    var_5 = bool('module.outer_function.inner_function' in var_0.doc)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_magic_name. Retrieved 8/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = '__str__'
    var_3 = 'Initializer'
    var_4 = 'String representation'
    var_5 = 'module'
    var_6 = 1
    var_7 = var_0.compile()
    var_8 = '__init__'
    var_9 = bool('__init__' not in var_7)
    assert var_9 is True
    var_10 = '__str__'
    var_11 = bool('__str__' not in var_7)
    assert var_11 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_func_api_with_kwonlyargs. Retrieved 10/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = None
    var_4 = 'kwarg'
    var_5 = [var_4, var_3]
    var_6 = [var_3]
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = '*'
    var_12 = bool('*' in var_0.doc['name'])
    assert var_12 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_attr'
    var_2 = 42
    var_3 = []
    var_4 = None
    var_5 = 0



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_true. Retrieved 12/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'posonly'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = module_0.Parser()
    var_8 = 'root'
    var_9 = 'func'
    var_10 = False
    var_11 = var_7.doc[var_9]
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_api_without_link. Retrieved 8/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    var_2 = 'test_node'
    var_3 = []
    var_4 = []
    var_5 = 'test_root'
    var_6 = ''
    var_7 = var_1.parse(var_5, var_6)
    var_8 = '\n<a id="{}"></a>'
    var_9 = bool('\n<a id="{}"></a>' not in var_1.doc['test_root.test_node'])
    assert var_9 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 8/15 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 13/30 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 12/32 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 13/34 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'root.A'
    var_5 = []
    var_6 = []
    var_7 = var_0.class_api(var_1, var_4, var_5, var_6)
    var_8 = 'root.A'
    var_9 = bool('root.A' in var_0.doc)
    assert var_9 is True
    var_10 = 'class A'
    var_11 = bool('class A' in var_0.doc['root.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A(enum.Enum): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'root.A'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = []
    var_8 = 'root.A'
    var_9 = bool('root.A' in var_0.doc)
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc['root.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A: x: int = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'A'
    var_5 = []
    var_6 = []
    var_7 = 'x'
    var_8 = 'int'
    var_9 = []
    var_10 = 1
    var_11 = []
    var_12 = []
    var_13 = 'root.A'
    var_14 = []
    var_15 = 'root.A'
    var_16 = bool('root.A' in var_0.doc)
    assert var_16 is True
    var_17 = 'Members'
    var_18 = bool('Members' in var_0.doc['root.A'])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A(enum.Enum): X = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'A'
    var_5 = 'enum.Enum'
    var_6 = []
    var_7 = []
    var_8 = 'X'
    var_9 = 1
    var_10 = []
    var_11 = None
    var_12 = []
    var_13 = 'root.A'
    var_14 = 'root.A'
    var_15 = bool('root.A' in var_0.doc)
    assert var_15 is True
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc['root.A'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class A: x: int = 1; del x'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'A'
    var_5 = []
    var_6 = []
    var_7 = 'x'
    var_8 = 'int'
    var_9 = []
    var_10 = 1
    var_11 = []
    var_12 = []
    var_13 = 'root.A'
    var_14 = []
    var_15 = 'root.A'
    var_16 = bool('root.A' in var_0.doc)
    assert var_16 is True
    var_17 = 'Members'
    var_18 = bool('Members' not in var_0.doc['root.A'])
    assert var_18 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_23_evaluates_to_false. Retrieved 8/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_class'
    var_3 = 'test_attr'
    var_4 = 42
    var_5 = []
    var_6 = None
    var_7 = 'object'
    var_8 = [var_7]
    var_9 = 'test_attr'
    var_10 = bool('test_attr' not in var_0.doc[var_2])
    assert var_10 is True



# Parsed testcases at query #42
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_enum_bases. Retrieved 7/11 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/17 statements.
# Partially parsed test_class_api_with_enums. Retrieved 9/19 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 9/20 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 9/16 statements.


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
    var_7 = []
    var_8 = 'enum.Enum'
    var_9 = bool('enum.Enum' in var_0.doc['test_module.A'])
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
    var_11 = bool('x' in var_0.doc['test_module.A'] and 'int' in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'X'
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 'test_module.A'
    var_9 = 'enum.Enum'
    var_10 = []
    var_11 = 'X'
    var_12 = bool('X' in var_0.doc['test_module.A'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'x'
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 'test_module.A'
    var_9 = []
    var_10 = 'x'
    var_11 = bool('x' not in var_0.doc['test_module.A'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class A: pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = '_x'
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 'test_module.A'
    var_9 = []
    var_10 = '_x'
    var_11 = bool('_x' not in var_0.doc['test_module.A'])
    assert var_11 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_visit_Attribute_with_typing. Retrieved 5/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'typing'
    var_1 = []
    var_2 = 'List'
    var_3 = []
    var_4 = 'root'
    var_5 = {}
    var_6 = module_0.Resolver(var_4, var_5)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_func_api_with_vararg. Retrieved 11/14 statements.


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
    var_12 = '*args'
    var_13 = bool('*args' in var_0.doc['func_name'])
    assert var_13 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_func_api_without_self. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 16/25 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = []
    var_4 = 'b'
    var_5 = []
    var_6 = 'c'
    var_7 = []
    var_8 = 'd'
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = '| a | b | c | d | return |'
    var_17 = bool('| a | b | c | d | return |' in var_0.doc['name'])
    assert var_17 is True
    var_18 = '|:---:|:---:|:---:|:---:|:---:|'
    var_19 = bool('|:---:|:---:|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_19 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = 'a'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = True
    var_13 = False
    var_14 = '| self | a | return |'
    var_15 = bool('| self | a | return |' in var_0.doc['name'])
    assert var_15 is True
    var_16 = '|:---:|:---:|:---:|'
    var_17 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = 'a'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = True
    var_13 = '| cls | a | return |'
    var_14 = bool('| cls | a | return |' in var_0.doc['name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|:---:|'
    var_16 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = 'args'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = '| a | *args | return |'
    var_14 = bool('| a | *args | return |' in var_0.doc['name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|:---:|'
    var_16 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'kwargs'
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = '| a | **kwargs | return |'
    var_14 = bool('| a | **kwargs | return |' in var_0.doc['name'])
    assert var_14 is True
    var_15 = '|:---:|:---:|:---:|'
    var_16 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
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
    var_11 = 2
    var_12 = []
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = '| a | b | return |'
    var_17 = bool('| a | b | return |' in var_0.doc['name'])
    assert var_17 is True
    var_18 = '|:---:|:---:|:---:|'
    var_19 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_19 is True
    var_20 = '| 1 | 2 |  |'
    var_21 = bool('| 1 | 2 |  |' in var_0.doc['name'])
    assert var_21 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root.int'
    var_2 = 'root.str'
    var_3 = 'int'
    var_4 = 'str'
    var_5 = []
    var_6 = 'a'
    var_7 = []
    var_8 = 'b'
    var_9 = []
    var_10 = None
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = 'root'
    var_15 = 'name'
    var_16 = 'None'
    var_17 = []
    var_18 = False
    var_19 = '| a | b | return |'
    var_20 = bool('| a | b | return |' in var_0.doc['name'])
    assert var_20 is True
    var_21 = '|:---:|:---:|:---:|'
    var_22 = bool('|:---:|:---:|:---:|' in var_0.doc['name'])
    assert var_22 is True
    var_23 = '| `int` | `str` | `None` |'
    var_24 = bool('| `int` | `str` | `None` |' in var_0.doc['name'])
    assert var_24 is True



