####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_m_single_name. Retrieved 1/2 statements.
# Partially parsed test_m_multiple_names. Retrieved 2/3 statements.
# Partially parsed test_m_three_names. Retrieved 3/4 statements.
# Partially parsed test_m_empty_string. Retrieved 1/2 statements.
# Partially parsed test_m_with_empty_strings. Retrieved 3/4 statements.
# Partially parsed test_m_multiple_empty_strings. Retrieved 2/3 statements.
# Failed to parse test_m_no_arguments.
# Partially parsed test_m_all_empty_strings. Retrieved 1/2 statements.
# Partially parsed test_m_single_empty_string. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'package'
    var_1 = 'module'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'pkg'
    var_1 = 'subpkg'
    var_2 = 'module'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = ''
    var_1 = [var_0]

def test_case_0():
    var_0 = 'package'
    var_1 = ''
    var_2 = 'module'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = ''
    var_1 = 'module'
    var_2 = [var_0, var_0, var_1, var_0]

def test_case_0():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]

def test_case_0():
    var_0 = ''
    var_1 = [var_0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_level. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_multiple_nested_levels. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = None
    var_1 = 'inner.value'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'attr'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None

def test_case_0():
    var_0 = 42
    var_1 = 'b.c.d.final'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_tuple_of_strs. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_floats. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_int_str. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_list_mixed_types. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_call_int. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_call_str. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_call_bool. Retrieved 3/8 statements.
# Partially parsed test_const_type_with_unknown_node. Retrieved 1/5 statements.
# Partially parsed test_const_type_with_list_with_none_element. Retrieved 2/7 statements.


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
    var_0 = True
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
    var_0 = 1.0
    var_1 = []
    var_2 = 2.0
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
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_tuple_of_strings. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_int_to_str. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_mixed_type_list. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_list_containing_non_constant. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_str. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_list. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unknown_call. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unsupported_node. Retrieved 1/4 statements.


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
    var_0 = True
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
    var_0 = 1
    var_1 = []
    var_2 = 2
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
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'list'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'unknown_func'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'variable'
    var_1 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/15 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 5/13 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/13 statements.
# Partially parsed test_globals_with_all_list. Retrieved 6/19 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 6/16 statements.
# Partially parsed test_globals_ignores_non_name_target. Retrieved 7/24 statements.
# Partially parsed test_globals_with_annassign_no_value. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 5
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.x']
    assert var_10 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'y'
    var_3 = 'hello'
    var_4 = []
    var_5 = None
    var_6 = 'test_module.y'
    var_7 = bool('test_module.y' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.y']
    assert var_8 == "'hello'"

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT'
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module.CONSTANT'
    var_7 = bool('test_module.CONSTANT' in var_0.root)
    assert var_7 is True
    var_8 = var_0.root['test_module.CONSTANT']
    var_9 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_9 is True
    var_10 = 'test_module.CONSTANT'
    var_11 = bool('test_module.CONSTANT' in var_0.const)
    assert var_11 is True
    var_12 = var_0.const['test_module.CONSTANT']
    assert var_12 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.func2'
    var_12 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'z'
    var_3 = 3.14
    var_4 = []
    var_5 = 'float'
    var_6 = 'test_module.z'
    var_7 = bool('test_module.z' in var_0.alias)
    assert var_7 is True
    var_8 = 'test_module.z'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = []
    var_6 = None
    var_7 = 'test_module.a'
    var_8 = bool('test_module.a' not in var_0.alias)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = []
    var_6 = 2
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.x'
    var_11 = bool('test_module.x' not in var_0.alias)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.x'
    var_8 = bool('test_module.x' not in var_0.alias)
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_table_basic. Retrieved 10/11 statements.
# Partially parsed test_table_single_column. Retrieved 7/8 statements.
# Partially parsed test_table_multiple_columns. Retrieved 13/14 statements.
# Partially parsed test_table_with_long_headers. Retrieved 10/11 statements.
# Partially parsed test_table_with_string_items. Retrieved 6/7 statements.
# Partially parsed test_table_mixed_items. Retrieved 8/9 statements.
# Partially parsed test_table_empty_items. Retrieved 4/5 statements.
# Partially parsed test_table_single_item. Retrieved 5/6 statements.
# Partially parsed test_table_wide_cell_content. Retrieved 7/8 statements.


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
    var_9 = [var_0, var_1]
    var_10 = '| a | b |\n|:---:|:---:|\n| c | d |\n| e | f |\n\n'

def test_case_0():
    var_0 = 'header'
    var_1 = 'row1'
    var_2 = [var_1]
    var_3 = 'row2'
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = [var_0]
    var_7 = '| header |\n|:---:|\n| row1 |\n| row2 |\n\n'

def test_case_0():
    var_0 = 'col1'
    var_1 = 'col2'
    var_2 = 'col3'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'd'
    var_8 = 'e'
    var_9 = 'f'
    var_10 = [var_7, var_8, var_9]
    var_11 = [var_6, var_10]
    var_12 = [var_0, var_1, var_2]
    var_13 = '| col1 | col2 | col3 |\n|:---:|:---:|:---:|\n| a | b | c |\n| d | e | f |\n\n'

def test_case_0():
    var_0 = 'header1'
    var_1 = 'header2'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = [var_2, var_3]
    var_5 = 'z'
    var_6 = 'w'
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_1]
    var_10 = '| header1 | header2 |\n|:-------:|:-------:|\n| x | y |\n| z | w |\n\n'

def test_case_0():
    var_0 = 'name'
    var_1 = 'value'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]
    var_5 = [var_0, var_1]
    var_6 = '| name | value |\n|:---:|:---:|\n| item1 |\n| item2 |\n\n'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = [var_2, var_3]
    var_5 = 'single'
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_1]
    var_8 = '| a | b |\n|:---:|:---:|\n| x | y |\n| single |\n\n'

def test_case_0():
    var_0 = 'header1'
    var_1 = 'header2'
    var_2 = []
    var_3 = [var_0, var_1]
    var_4 = '| header1 | header2 |\n|:---:|:---:|\n\n'

def test_case_0():
    var_0 = 'col'
    var_1 = 'value'
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = '| col |\n|:---:|\n| value |\n\n'

def test_case_0():
    var_0 = 'short'
    var_1 = 'verylongheader'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = [var_0, var_1]
    var_7 = '| short | verylongheader |\n|:---:|:-----:|\n| a | b |\n\n'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/15 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_assigned_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 10/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass(BaseClass):\n    pass\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestEnum(Enum):\n    MEMBER1: int\n    MEMBER2: str\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestEnum'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestEnum'
    var_11 = bool('test_module.TestEnum' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr1\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass EmptyClass:\n    pass\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.EmptyClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.EmptyClass'
    var_11 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass TestClass:\n    value = 42\n    name = 'test'\n"
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass MultiBase(Base1, Base2):\n    pass\n'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.MultiBase'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.MultiBase'
    var_11 = bool('test_module.MultiBase' in var_0.doc)
    assert var_11 is True
    var_12 = 'Bases'
    var_13 = bool('Bases' in var_0.doc['test_module.MultiBase'])
    assert var_13 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 7/10 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/15 statements.
# Partially parsed test_class_api_with_enums. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/14 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 8/17 statements.
# Partially parsed test_class_api_empty_class. Retrieved 7/8 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/13 statements.
# Partially parsed test_class_api_with_multiple_members. Retrieved 11/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class MyClass(BaseClass): pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'BaseClass'
    var_5 = []
    var_6 = []
    var_7 = 'test_module.MyClass'
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc['test_module.MyClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = 'test_module.MyClass'
    var_12 = bool('test_module.MyClass' in var_0.doc)
    assert var_12 is True
    var_13 = 'Members'
    var_14 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = 'RED'
    var_5 = []
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.Color'
    var_12 = 'test_module.Color'
    var_13 = bool('test_module.Color' in var_0.doc)
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 5
    var_7 = []
    var_8 = 1
    var_9 = 'test_module'
    var_10 = 'test_module.MyClass'
    var_11 = 'test_module.MyClass'
    var_12 = bool('test_module.MyClass' in var_0.doc)
    assert var_12 is True
    var_13 = bool('Members' not in var_0.doc['test_module.MyClass'] or '_private' not in var_0.doc['test_module.MyClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass'
    var_12 = 'test_module.MyClass'
    var_13 = bool('test_module.MyClass' in var_0.doc)
    assert var_13 is True
    var_14 = 'Members'
    var_15 = bool('Members' not in var_0.doc['test_module.MyClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True
    var_8 = var_0.doc[var_4]

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'int'
    var_7 = 'test_module'
    var_8 = 'test_module.MyClass'
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_12 is True
    var_13 = 'int'
    var_14 = bool('int' in var_0.doc['test_module.MyClass'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = 'member2'
    var_10 = []
    var_11 = 'str'
    var_12 = []
    var_13 = 'test'
    var_14 = []
    var_15 = 'test_module'
    var_16 = 'test_module.MyClass'
    var_17 = 'test_module.MyClass'
    var_18 = bool('test_module.MyClass' in var_0.doc)
    assert var_18 is True
    var_19 = 'Members'
    var_20 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_20 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/20 statements.
# Partially parsed test_class_api_with_enums. Retrieved 7/23 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 6/21 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/20 statements.
# Partially parsed test_class_api_empty. Retrieved 6/10 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 6/16 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'TestClass'
    var_6 = 'TestClass'
    var_7 = bool('TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Base'
    var_9 = bool('Base' in var_0.doc['TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'str'
    var_4 = []
    var_5 = 'test'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'TestClass'
    var_10 = 'TestClass'
    var_11 = bool('TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'member1'
    var_13 = bool('member1' in var_0.doc['TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER'
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'TestEnum'
    var_10 = 'TestEnum'
    var_11 = bool('TestEnum' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['TestEnum'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'test'
    var_4 = []
    var_5 = 'test_module'
    var_6 = 'TestClass'
    var_7 = 'TestClass'
    var_8 = bool('TestClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'member1'
    var_10 = bool('member1' not in var_0.doc['TestClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'int'
    var_4 = []
    var_5 = 5
    var_6 = []
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'TestClass'
    var_10 = 'TestClass'
    var_11 = bool('TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = '_private'
    var_13 = bool('_private' not in var_0.doc['TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'TestClass'
    var_7 = bool('TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = var_0.doc['TestClass']
    assert var_8 == '## class TestClass\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'TestClass'
    var_8 = 'TestClass'
    var_9 = bool('TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Base1'
    var_11 = bool('Base1' in var_0.doc['TestClass'])
    assert var_11 is True
    var_12 = 'Base2'
    var_13 = bool('Base2' in var_0.doc['TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'TestClass'
    var_8 = 'TestClass'
    var_9 = bool('TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'member1'
    var_11 = bool('member1' in var_0.doc['TestClass'])
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0].lineno
    assert var_6 == 1

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1\ny = 2\nz = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if False:\n        x = 1\n    else:\n        y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 4

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    x = 1\n    y = 2\nelse:\n    z = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept TypeError:\n    z = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x = 1\nif True:\n    y = 2\nz = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if True:\n        if True:\n            x = 1'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #11
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = bool(var_2 == [' ', ' ', ' '])
    assert var_3 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '1'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = bool(var_6 == ['`1`'])
    assert var_7 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.value
    var_5 = None
    var_6 = [var_5, var_4, var_5]
    var_7 = module_1._defaults(var_6)
    var_8 = bool(var_7 == [' ', '`42`', ' '])
    assert var_8 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a & b'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = bool(var_6 == ['<code>a & b</code>'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_class_api_enums_predicate. Retrieved 11/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Color(enum.Enum):\n    RED: int = 1\n    GREEN: int = 2\n    BLUE: int = 3\n'
    var_2 = True
    var_3 = module_1.parse(var_1, type_comments=var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.Color'
    var_8 = 'Enum'
    var_9 = None
    var_10 = []
    var_11 = var_5.body
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc[var_7])
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_globals_predicate_line_18_false. Retrieved 9/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 18 evaluates to False when len(node.targets) != 1'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'x'
    var_4 = None
    var_5 = []
    var_6 = 'y'
    var_7 = []
    var_8 = 42
    var_9 = []
    var_10 = var_1.alias
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '42'
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = "'hello'"
    var_5 = module_0.parse(var_4, mode=var_1)
    var_6 = var_5.body
    var_7 = [var_3, var_6]
    var_8 = module_1._defaults(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = '`42`'
    var_12 = bool('`42`' in var_9[0])
    assert var_12 is True
    var_13 = "`'hello'`"
    var_14 = bool("`'hello'`" in var_9[1])
    assert var_14 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '100'
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = None
    var_5 = [var_4, var_3, var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = var_7[0]
    assert var_8 == ' '
    var_9 = '`100`'
    var_10 = bool('`100`' in var_7[1])
    assert var_10 is True
    var_11 = var_7[2]
    assert var_11 == ' '

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "'a|b'"
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = [var_3]
    var_5 = module_1._defaults(var_4)
    var_6 = list(var_5)
    var_7 = '&#124;'
    var_8 = bool('&#124;' in var_6[0])
    assert var_8 is True
    var_9 = '<code>'
    var_10 = bool('<code>' in var_6[0])
    assert var_10 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = "'a&b'"
    var_1 = 'eval'
    var_2 = module_0.parse(var_0, mode=var_1)
    var_3 = var_2.body
    var_4 = [var_3]
    var_5 = module_1._defaults(var_4)
    var_6 = list(var_5)
    var_7 = '<code>'
    var_8 = bool('<code>' in var_6[0])
    assert var_8 is True

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
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' '])
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_globals_predicate_line_18. Retrieved 5/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_public_with_root_module. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_public_submodule. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_matching. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_not_matching. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_all_list_empty. Retrieved 3/5 statements.
# Partially parsed test_is_public_module_in_imp_with_public_children. Retrieved 3/6 statements.
# Partially parsed test_is_public_module_in_imp_without_public_children. Retrieved 3/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_all_list_parent_match. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.submodule'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule._private'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.other'
    var_2 = 'mymodule.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.public_func'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.__init__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule.submodule'
    var_2 = 'mymodule.submodule.func'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_attr_single_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_attribute_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_attr_multiple_levels_with_valid_path. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = None
    var_1 = 'inner.value'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = None
    var_1 = 'attr'

def test_case_0():
    var_0 = 42
    var_1 = 'b.c.prop'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 7/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 'test_class'
    var_9 = []
    var_10 = var_0.doc['test_class']
    assert var_10 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_parser_new_class_method. Retrieved 3/5 statements.
# Partially parsed test_parser_new_class_method_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_public_predicate_line_5_false. Retrieved 8/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 5 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = 'mymodule'
    var_3 = 'mymodule.submodule'
    var_4 = {var_3}
    var_5 = 'mymodule.other'
    var_6 = 'doc'
    var_7 = var_1.is_public(var_2)
    assert var_7 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_compile_basic. Retrieved 4/9 statements.
# Partially parsed test_compile_with_toc. Retrieved 5/10 statements.
# Partially parsed test_compile_with_link. Retrieved 4/9 statements.
# Partially parsed test_compile_multiple_names. Retrieved 4/13 statements.
# Partially parsed test_compile_magic_method_skipped. Retrieved 4/12 statements.
# Partially parsed test_compile_private_name_excluded. Retrieved 4/13 statements.
# Partially parsed test_compile_with_constants. Retrieved 5/11 statements.
# Partially parsed test_compile_sorted_by_level_and_name. Retrieved 6/18 statements.
# Partially parsed test_compile_with_toc_and_link. Retrieved 3/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = '# Module `module`'
    var_5 = bool('# Module `module`' in var_3)
    assert var_5 is True
    var_6 = 'Module docstring'
    var_7 = bool('Module docstring' in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '**Table of contents:**'
    var_6 = bool('**Table of contents:**' in var_4)
    assert var_6 is True
    var_7 = 'Module docstring'
    var_8 = bool('Module docstring' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()
    var_4 = 'Module docstring'
    var_5 = bool('Module docstring' in var_3)
    assert var_5 is True
    var_6 = '<a id="module"></a>'
    var_7 = bool('<a id="module"></a>' in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'Module docstring'
    var_5 = bool('Module docstring' in var_3)
    assert var_5 is True
    var_6 = 'Function docstring'
    var_7 = bool('Function docstring' in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'Module docstring'
    var_5 = bool('Module docstring' in var_3)
    assert var_5 is True
    var_6 = '__init__'
    var_7 = bool('__init__' not in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'Module docstring'
    var_5 = bool('Module docstring' in var_3)
    assert var_5 is True
    var_6 = 'Private docstring'
    var_7 = bool('Private docstring' not in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'module'
    var_4 = var_2.compile()
    var_5 = 'Module docstring'
    var_6 = bool('Module docstring' in var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'A docstring'
    var_5 = 'Z docstring'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    assert var_3 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0, var_0)
    var_2 = var_1.compile()
    var_3 = '**Table of contents:**'
    var_4 = bool('**Table of contents:**' in var_2)
    assert var_4 is True
    var_5 = '<a id="module"></a>'
    var_6 = bool('<a id="module"></a>' in var_2)
    assert var_6 is True
    var_7 = 'Module docstring'
    var_8 = bool('Module docstring' in var_2)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_class_api_node_type_comment_not_none. Retrieved 5/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr_name'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'int'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_class_api_predicate_line_25_false. Retrieved 12/27 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that predicate at line 25 (is_public_family(attr)) evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '_private_attr'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = {}
    var_8 = False
    var_9 = 0
    var_10 = var_2.id
    assert var_10 == '_private_attr'
    var_11 = False
    var_12 = True
    assert var_12 is False
    var_13 = len(var_7)
    assert var_13 == 0



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_func_api_simple_function. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 12/18 statements.
# Partially parsed test_func_api_no_return_type. Retrieved 12/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def foo(x: int, y: str) -> bool: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.foo'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.foo'
    var_13 = bool('test_module.foo' in var_0.doc)
    assert var_13 is True
    var_14 = '|'
    var_15 = bool('|' in var_0.doc['test_module.foo'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = "def bar(x: int = 5, y: str = 'hello') -> None: pass"
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.bar'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.bar'
    var_13 = bool('test_module.bar' in var_0.doc)
    assert var_13 is True
    var_14 = '|'
    var_15 = bool('|' in var_0.doc['test_module.bar'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def method(self, x: int) -> str: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.MyClass.method'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = True
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.MyClass.method'
    var_13 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_13 is True
    var_14 = 'Self'
    var_15 = bool('Self' in var_0.doc['test_module.MyClass.method'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def method(cls, x: int) -> str: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.MyClass.method'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = True
    var_10 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_9)
    var_11 = 'test_module.MyClass.method'
    var_12 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_12 is True
    var_13 = 'type[Self]'
    var_14 = bool('type[Self]' in var_0.doc['test_module.MyClass.method'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def func(*args: int, **kwargs: str) -> None: pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.func'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '|'
    var_15 = bool('|' in var_0.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = "def func(x: int, *, y: str = 'test') -> bool: pass"
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.func'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '|'
    var_15 = bool('|' in var_0.doc['test_module.func'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def func(x): pass'
    var_4 = module_1.parse(var_3)
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.func'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_11 = var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = 'x'
    var_15 = bool('x' in var_0.doc['test_module.func'])
    assert var_15 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.link
    assert var_3 is True
    var_4 = var_2.toc
    assert var_4 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_with_different_b_level. Retrieved 7/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n'''Module docstring.'''\nimport os\nx = 1\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.level)
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.imp)
    assert var_9 is True
    var_10 = var_0.level['test_module']
    assert var_10 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom typing import List\n'
    var_2 = 'pkg.module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'pkg.module'
    var_5 = bool('pkg.module' in var_0.imp)
    assert var_5 is True
    var_6 = var_0.alias
    var_7 = len(var_6)
    var_8 = bool(var_7 >= 2)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef foo():\n    '''Function docstring.'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass MyClass:\n    '''Class docstring.'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nclass Outer:\n    '''Outer class.'''\n    class Inner:\n        '''Inner class.'''\n        pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.Outer'
    var_5 = bool('test_module.Outer' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.Outer.Inner'
    var_7 = bool('test_module.Outer.Inner' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n'''This is a module docstring.'''\nx = 1\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = 'module docstring'
    var_7 = bool('module docstring' in var_0.docstring['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.CONSTANT'
    var_5 = bool('test_module.CONSTANT' in var_0.const)
    assert var_5 is True
    var_6 = 'test_module.ANOTHER_CONST'
    var_7 = bool('test_module.ANOTHER_CONST' in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = "\ndef foo():\n    '''Function.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module'
    var_6 = bool('test_module' in var_1.doc)
    assert var_6 is True
    var_7 = '<a id='
    var_8 = bool('<a id=' in var_1.doc['test_module'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Parser(b_level=var_0)
    var_2 = "\n'''Module.'''\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = var_1.doc[var_3]
    var_6 = '###'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport sys\n'
    var_2 = 'pkg.subpkg.module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.level['pkg.subpkg.module']
    assert var_4 == 2

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nasync def async_func():\n    '''Async function.'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.async_func'
    var_5 = bool('test_module.async_func' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n@property\ndef prop():\n    '''Property.'''\n    return 42\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.prop'
    var_5 = bool('test_module.prop' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'test_module.bar'
    var_7 = bool('test_module.bar' in var_0.imp['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.func'
    var_5 = bool('test_module.func' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\nif True:\n    def conditional_func():\n        '''Conditional function.'''\n        pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.conditional_func'
    var_5 = bool('test_module.conditional_func' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.in_try'
    var_5 = bool('test_module.in_try' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef func1():\n    '''First function.'''\n    pass\n"
    var_2 = "\ndef func2():\n    '''Second function.'''\n    pass\n"
    var_3 = 'module1'
    var_4 = var_0.parse(var_3, var_1)
    var_5 = 'module2'
    var_6 = var_0.parse(var_5, var_2)
    var_7 = 'module1.func1'
    var_8 = bool('module1.func1' in var_0.doc)
    assert var_8 is True
    var_9 = 'module2.func2'
    var_10 = bool('module2.func2' in var_0.doc)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\ndef example():\n    '''Example function.\n    \n    >>> x = 1\n    >>> print(x)\n    1\n    '''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.example'
    var_5 = bool('test_module.example' in var_0.docstring)
    assert var_5 is True
    var_6 = '```python'
    var_7 = bool('```python' in var_0.docstring['test_module.example'])
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_visit_name_with_self_ty. Retrieved 5/11 statements.
# Partially parsed test_visit_name_with_alias_replacement. Retrieved 7/11 statements.
# Partially parsed test_visit_name_with_typevar_in_alias. Retrieved 9/13 statements.
# Partially parsed test_visit_name_without_alias. Retrieved 5/9 statements.
# Partially parsed test_visit_name_with_circular_alias. Retrieved 6/10 statements.
# Partially parsed test_visit_name_with_complex_alias. Retrieved 7/13 statements.
# Partially parsed test_visit_name_with_empty_root. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name replaces self_ty with Self.'
    var_1 = 'module'
    var_2 = {}
    var_3 = 'MyType'
    var_4 = module_0.Resolver(var_1, var_2, var_3)
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name replaces name with alias expression.'
    var_1 = 'module'
    var_2 = 'module.MyAlias'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 'MyAlias'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name returns original node for TypeVar.'
    var_1 = 'module'
    var_2 = 'module.T'
    var_3 = 'module.TypeVar'
    var_4 = "TypeVar('T')"
    var_5 = 'typing.TypeVar'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.Resolver(var_1, var_6)
    var_8 = 'T'
    var_9 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name returns original node when no alias exists.'
    var_1 = 'module'
    var_2 = {}
    var_3 = module_0.Resolver(var_1, var_2)
    var_4 = 'SomeName'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name skips circular alias references.'
    var_1 = 'module'
    var_2 = 'module.A'
    var_3 = {var_2: var_2}
    var_4 = module_0.Resolver(var_1, var_3)
    var_5 = 'A'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name with complex alias expression.'
    var_1 = 'module'
    var_2 = 'module.Complex'
    var_3 = 'List[int]'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = 'Complex'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test visit_Name with empty root module.'
    var_1 = ''
    var_2 = 'MyName'
    var_3 = 'str'
    var_4 = {var_2: var_3}
    var_5 = module_0.Resolver(var_1, var_4)
    var_6 = []



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_empty_sequence. Retrieved 1/3 statements.
# Partially parsed test_e_type_none_in_elements. Retrieved 2/4 statements.
# Partially parsed test_e_type_single_constant_int. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_constant_str. Retrieved 1/5 statements.
# Partially parsed test_e_type_multiple_same_type. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 2/8 statements.
# Partially parsed test_e_type_mixed_types_in_sequence. Retrieved 2/7 statements.
# Partially parsed test_e_type_mixed_types_different_sequences. Retrieved 2/8 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 2/6 statements.
# Partially parsed test_e_type_multiple_sequences_with_mixed_types. Retrieved 4/12 statements.
# Partially parsed test_e_type_single_sequence_multiple_constants. Retrieved 3/9 statements.
# Partially parsed test_e_type_float_constant. Retrieved 1/5 statements.
# Partially parsed test_e_type_bool_constant. Retrieved 1/5 statements.
# Partially parsed test_e_type_none_constant. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'hello'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'hello'
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = None
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
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1.5
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = None
    var_1 = []



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_imports_with_import_statement. Retrieved 4/9 statements.
# Partially parsed test_imports_with_import_as_alias. Retrieved 4/9 statements.
# Partially parsed test_imports_with_import_multiple_names. Retrieved 5/11 statements.
# Partially parsed test_imports_with_from_import_absolute. Retrieved 6/11 statements.
# Partially parsed test_imports_with_from_import_as_alias. Retrieved 6/11 statements.
# Partially parsed test_imports_with_from_import_relative_level_1. Retrieved 6/11 statements.
# Partially parsed test_imports_with_from_import_relative_level_2. Retrieved 6/11 statements.
# Partially parsed test_imports_with_from_import_no_module. Retrieved 5/10 statements.
# Partially parsed test_imports_with_from_import_multiple_names. Retrieved 7/13 statements.
# Partially parsed test_imports_with_from_import_star. Retrieved 6/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'mymodule'
    var_4 = var_0.alias['mymodule.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'mymodule'
    var_4 = var_0.alias['mymodule.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = None
    var_3 = 'sys'
    var_4 = 'mymodule'
    var_5 = var_0.alias['mymodule.os']
    assert var_5 == 'os'
    var_6 = var_0.alias['mymodule.sys']
    assert var_6 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = var_0.alias['mymodule.path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'p'
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = var_0.alias['mymodule.p']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'submodule'
    var_2 = 'func'
    var_3 = None
    var_4 = 1
    var_5 = 'pkg.mymodule'
    var_6 = var_0.alias['pkg.mymodule.func']
    assert var_6 == 'pkg.submodule.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'other'
    var_2 = 'Class'
    var_3 = None
    var_4 = 2
    var_5 = 'pkg.sub.mymodule'
    var_6 = var_0.alias['pkg.sub.mymodule.Class']
    assert var_6 == 'pkg.other.Class'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = None
    var_2 = 'func'
    var_3 = 1
    var_4 = 'pkg.mymodule'
    var_5 = var_0.alias['pkg.mymodule.func']
    assert var_5 == 'pkg.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 'getcwd'
    var_5 = 0
    var_6 = 'mymodule'
    var_7 = var_0.alias['mymodule.path']
    assert var_7 == 'os.path'
    var_8 = var_0.alias['mymodule.getcwd']
    assert var_8 == 'os.getcwd'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = '*'
    var_3 = None
    var_4 = 0
    var_5 = 'mymodule'
    var_6 = var_0.alias['mymodule.*']
    assert var_6 == 'os.*'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_compile_basic. Retrieved 5/12 statements.
# Partially parsed test_compile_with_toc. Retrieved 6/13 statements.
# Partially parsed test_compile_with_link. Retrieved 5/12 statements.
# Partially parsed test_compile_multiple_items. Retrieved 5/16 statements.
# Partially parsed test_compile_magic_method_skipped. Retrieved 5/15 statements.
# Partially parsed test_compile_with_constants. Retrieved 5/14 statements.
# Partially parsed test_compile_empty. Retrieved 5/12 statements.
# Partially parsed test_compile_with_nested_items. Retrieved 6/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with basic parser setup.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '# Module `test`'
    var_6 = bool('# Module `test`' in var_4)
    assert var_6 is True
    var_7 = 'Test module'
    var_8 = bool('Test module' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with table of contents enabled.'
    var_1 = False
    var_2 = 1
    var_3 = True
    var_4 = module_0.Parser(var_1, var_2, var_3)
    var_5 = var_4.compile()
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True
    var_8 = '# Module `test`'
    var_9 = bool('# Module `test`' in var_5)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with link enabled.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '<a id="test"></a>'
    var_6 = bool('<a id="test"></a>' in var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with multiple documentation items.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '# Module `test`'
    var_6 = bool('# Module `test`' in var_4)
    assert var_6 is True
    var_7 = '## func()'
    var_8 = bool('## func()' in var_4)
    assert var_8 is True
    var_9 = 'Test module'
    var_10 = bool('Test module' in var_4)
    assert var_10 is True
    var_11 = 'Test function'
    var_12 = bool('Test function' in var_4)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that magic methods without docstring are skipped.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '# Module `test`'
    var_6 = bool('# Module `test`' in var_4)
    assert var_6 is True
    var_7 = '__init__'
    var_8 = bool('__init__' not in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with constants table.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = 'Constants'
    var_6 = bool('Constants' in var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with empty parser.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    assert var_4 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test compile with nested documentation items.'
    var_1 = False
    var_2 = 1
    var_3 = True
    var_4 = module_0.Parser(var_1, var_2, var_3)
    var_5 = var_4.compile()
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True
    var_8 = 'pkg'
    var_9 = bool('pkg' in var_5)
    assert var_9 is True
    var_10 = 'pkg.mod'
    var_11 = bool('pkg.mod' in var_5)
    assert var_11 is True
    var_12 = 'pkg.mod.func'
    var_13 = bool('pkg.mod.func' in var_5)
    assert var_13 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 5/23 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'int'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_globals_const_predicate_false. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 33 evaluates to False when const already has a value.'
    var_1 = module_0.Parser()
    var_2 = 'MY_CONST'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'test_module'
    var_8 = var_1.const['test_module.MY_CONST']
    assert var_8 == 'str'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/12 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_enum. Retrieved 7/22 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 6/20 statements.
# Partially parsed test_class_api_with_private_member. Retrieved 8/18 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/17 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'MyClass'
    var_6 = 'MyClass'
    var_7 = bool('MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'int'
    var_3 = []
    var_4 = 'member_var'
    var_5 = None
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'MyClass'
    var_9 = 'MyClass'
    var_10 = bool('MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'member_var'
    var_12 = bool('member_var' in var_0.doc['MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'RED'
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'MyEnum'
    var_10 = 'MyEnum'
    var_11 = bool('MyEnum' in var_0.doc)
    assert var_11 is True
    var_12 = 'RED'
    var_13 = bool('RED' in var_0.doc['MyEnum'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'temp_var'
    var_3 = 42
    var_4 = []
    var_5 = 'test_module'
    var_6 = 'MyClass'
    var_7 = 'MyClass'
    var_8 = bool('MyClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'temp_var'
    var_10 = bool('temp_var' not in var_0.doc['MyClass'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'str'
    var_3 = []
    var_4 = '_private_var'
    var_5 = None
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'MyClass'
    var_9 = 'MyClass'
    var_10 = bool('MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = '_private_var'
    var_12 = bool('_private_var' not in var_0.doc['MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'value'
    var_3 = 10
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'MyClass'
    var_8 = 'MyClass'
    var_9 = bool('MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'value'
    var_11 = bool('value' in var_0.doc['MyClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'EmptyClass'
    var_7 = bool('EmptyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'MyClass'
    var_8 = 'MyClass'
    var_9 = bool('MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Base1'
    var_11 = bool('Base1' in var_0.doc['MyClass'])
    assert var_11 is True
    var_12 = 'Base2'
    var_13 = bool('Base2' in var_0.doc['MyClass'])
    assert var_13 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_docstring. Retrieved 2/2 statements.
# Partially parsed test_load_docstring_with_doctest_formatting. Retrieved 2/13 statements.
# Partially parsed test_load_docstring_missing_attribute. Retrieved 2/9 statements.
# Partially parsed test_load_docstring_nested_attribute. Retrieved 2/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a test function docstring.'
    var_1 = module_0.Parser()
    var_2 = 'test_module.TestClass'
    var_3 = bool('test_module.TestClass' in var_1.docstring)
    assert var_3 is True
    var_4 = 'This is a test class docstring.'
    var_5 = bool('This is a test class docstring.' in var_1.docstring['test_module.TestClass'])
    assert var_5 is True
    var_6 = 'test_module.test_function'
    var_7 = bool('test_module.test_function' in var_1.docstring)
    assert var_7 is True
    var_8 = 'This is a test function docstring.'
    var_9 = bool('This is a test function docstring.' in var_1.docstring['test_module.test_function'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a test function docstring.'
    var_1 = module_0.Parser()
    var_2 = 'test_module.TestClass'
    var_3 = bool('test_module.TestClass' in var_1.docstring)
    assert var_3 is True
    var_4 = 'This is a test class docstring.'
    var_5 = bool('This is a test class docstring.' in var_1.docstring['test_module.TestClass'])
    assert var_5 is True
    var_6 = 'test_module.test_function'
    var_7 = bool('test_module.test_function' in var_1.docstring)
    assert var_7 is True
    var_8 = 'This is a test function docstring.'
    var_9 = bool('This is a test function docstring.' in var_1.docstring['test_module.test_function'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.TestClassWithDoctest'
    var_3 = bool('test_module.TestClassWithDoctest' in var_1.docstring)
    assert var_3 is True
    var_4 = '```python'
    var_5 = bool('```python' in var_1.docstring['test_module.TestClassWithDoctest'])
    assert var_5 is True
    var_6 = '>>> x = 1'
    var_7 = bool('>>> x = 1' in var_1.docstring['test_module.TestClassWithDoctest'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.NonExistent'
    var_3 = bool('test_module.NonExistent' not in var_1.docstring)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
    var_2 = 'test_module.OuterClass'
    var_3 = bool('test_module.OuterClass' in var_1.docstring)
    assert var_3 is True
    var_4 = 'Outer class docstring.'
    var_5 = bool('Outer class docstring.' in var_1.docstring['test_module.OuterClass'])
    assert var_5 is True
    var_6 = 'test_module.OuterClass.InnerClass'
    var_7 = bool('test_module.OuterClass.InnerClass' in var_1.docstring)
    assert var_7 is True
    var_8 = 'Inner class docstring.'
    var_9 = bool('Inner class docstring.' in var_1.docstring['test_module.OuterClass.InnerClass'])
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_with_string_constant. Retrieved 4/12 statements.
# Partially parsed test_globals_ignores_lowercase_assignment. Retrieved 4/12 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/13 statements.
# Partially parsed test_globals_with_multiple_targets_ignored. Retrieved 6/14 statements.
# Partially parsed test_globals_with_list_constant. Retrieved 4/12 statements.
# Partially parsed test_globals_annotated_without_value_ignored. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 42'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.x']
    assert var_6 == '42'
    var_7 = var_0.const['test_module.x']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MY_CONSTANT = 100'
    var_3 = 0
    var_4 = 'test_module.MY_CONSTANT'
    var_5 = bool('test_module.MY_CONSTANT' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.MY_CONSTANT']
    assert var_6 == '100'
    var_7 = 'test_module.MY_CONSTANT'
    var_8 = bool('test_module.MY_CONSTANT' in var_0.root)
    assert var_8 is True
    var_9 = var_0.const['test_module.MY_CONSTANT']
    assert var_9 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = 0
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ('ClassA', 'ClassB')"
    var_4 = 0
    var_5 = 'test_module.ClassA'
    var_6 = bool('test_module.ClassA' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.ClassB'
    var_8 = bool('test_module.ClassB' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "MESSAGE = 'hello'"
    var_3 = 0
    var_4 = 'test_module.MESSAGE'
    var_5 = bool('test_module.MESSAGE' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.const['test_module.MESSAGE']
    assert var_6 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'variable = 5'
    var_3 = 0
    var_4 = 'test_module.variable'
    var_5 = bool('test_module.variable' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.variable'
    var_7 = bool('test_module.variable' not in var_0.root)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'FLAG = True  # type: bool'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.FLAG'
    var_6 = bool('test_module.FLAG' in var_0.const)
    assert var_6 is True
    var_7 = var_0.const['test_module.FLAG']
    assert var_7 == 'bool'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a = b = 10'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'ITEMS = [1, 2, 3]'
    var_3 = 0
    var_4 = 'test_module.ITEMS'
    var_5 = bool('test_module.ITEMS' in var_0.const)
    assert var_5 is True
    var_6 = var_0.const['test_module.ITEMS']
    assert var_6 == 'list[int]'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int'
    var_3 = 0
    var_4 = var_0.alias
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_func_api_predicate_line_32_false. Retrieved 11/59 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'default_value'
    var_7 = [var_6]
    var_8 = 'test_module'
    var_9 = 'test_func'
    var_10 = False
    var_11 = var_0.predicate_result
    assert var_11 is False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.test_func'
    var_10 = False
    var_11 = 'test_module.test_func'
    var_12 = bool('test_module.test_func' in var_0.doc)
    assert var_12 is True
    var_13 = '/'
    var_14 = bool('/' in var_0.doc['test_module.test_func'])
    assert var_14 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_members. Retrieved 8/19 statements.
# Partially parsed test_class_api_with_enums. Retrieved 8/25 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 10/26 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 8/23 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/10 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_multiple_bases. Retrieved 6/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Bases'
    var_9 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'str'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'test_module.TestClass'
    var_10 = bool('test_module.TestClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'VALUE1'
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestEnum'
    var_12 = 'test_module.TestEnum'
    var_13 = bool('test_module.TestEnum' in var_0.doc)
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc['test_module.TestEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'str'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'public'
    var_8 = 'int'
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.TestClass'
    var_12 = 'test_module.TestClass'
    var_13 = bool('test_module.TestClass' in var_0.doc)
    assert var_13 is True
    var_14 = 'Members'
    var_15 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'str'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = 'test_module.TestClass'
    var_10 = bool('test_module.TestClass' in var_0.doc)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.TestClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_11 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_visit_constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_invalid_syntax_string. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_valid_name_string. Retrieved 5/8 statements.
# Partially parsed test_visit_constant_with_self_type_string. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_complex_expression_string. Retrieved 4/7 statements.
# Partially parsed test_visit_constant_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_empty_string. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_with_boolean_value. Retrieved 4/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not valid python @@@'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.int'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'list[int]'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = None
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = ''
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = True
    var_4 = []



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_predicate_line_25_false. Retrieved 8/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (is_public_family(attr)) evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '_private_attr'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_1.doc)
    assert var_11 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'test_module'
    var_4 = 'test_module.operating_system'
    var_5 = bool('test_module.operating_system' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.operating_system']
    assert var_6 == 'os'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_regular_assignment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 9/18 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/15 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 9/18 statements.
# Partially parsed test_globals_with_annotated_no_value. Retrieved 9/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int = 5'
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.x']
    assert var_7 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'y = 10'
    var_4 = 0
    var_5 = 'test_module.y'
    var_6 = bool('test_module.y' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.y']
    assert var_7 == '10'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MAX_SIZE = 100'
    var_4 = 0
    var_5 = 'test_module.MAX_SIZE'
    var_6 = bool('test_module.MAX_SIZE' in var_0.root)
    assert var_6 is True
    var_7 = var_0.root['test_module.MAX_SIZE']
    var_8 = bool(var_0.root['test_module.MAX_SIZE'] == var_1)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = 0
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ('item1', 'item2')"
    var_4 = 0
    var_5 = 'test_module.item1'
    var_6 = bool('test_module.item1' in var_0.imp[var_1])
    assert var_6 is True
    var_7 = 'test_module.item2'
    var_8 = bool('test_module.item2' in var_0.imp[var_1])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a = b = 5'
    var_4 = 0
    var_5 = var_0.alias
    var_6 = len(var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'z = 42  # type: int'
    var_4 = True
    var_5 = 0
    var_6 = 'test_module.z'
    var_7 = bool('test_module.z' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.z']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '[a, b] = [1, 2]'
    var_4 = 0
    var_5 = var_0.alias
    var_6 = len(var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'var: str'
    var_4 = 0
    var_5 = var_0.alias
    var_6 = len(var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 10/28 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True.'
    var_1 = 'test_module'
    var_2 = 'MyType'
    var_3 = f'{var_1}.{var_2}'
    var_4 = 'int'
    var_5 = {var_3: var_4}
    var_6 = ''
    var_7 = module_0.Resolver(var_1, var_5, var_6)
    var_8 = []
    var_9 = None
    var_10 = '_m'



# Parsed testcases at query #45
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.path.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__init__.function'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__main__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.public.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public._private.name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.name._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '__name__.public.module'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__magic__.another_public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '___name'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'collections.abc.Mapping'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'collections._abc.Mapping'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 6/32 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'int'
    var_7 = 'test_module'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/12 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 7/22 statements.
# Partially parsed test_class_api_with_class_members. Retrieved 8/19 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/21 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 8/19 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/17 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = 'test_module.TestClass'
    var_7 = bool('test_module.TestClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['test_module.TestClass'])
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER1'
    var_6 = 1
    var_7 = []
    var_8 = 'test_module'
    var_9 = 'test_module.TestEnum'
    var_10 = 'test_module.TestEnum'
    var_11 = bool('test_module.TestEnum' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['test_module.TestEnum'])
    assert var_13 is True
    var_14 = 'MEMBER1'
    var_15 = bool('MEMBER1' in var_0.doc['test_module.TestEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'str'
    var_4 = []
    var_5 = 'default'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = 'attr1'
    var_15 = bool('attr1' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 1
    var_4 = []
    var_5 = None
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Members'
    var_11 = bool('Members' not in var_0.doc['test_module.TestClass'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private_attr'
    var_3 = 'str'
    var_4 = []
    var_5 = 'private'
    var_6 = []
    var_7 = 1
    var_8 = 'test_module'
    var_9 = 'test_module.TestClass'
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = '_private_attr'
    var_13 = bool('_private_attr' not in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = 'test_module.TestClass'
    var_9 = bool('test_module.TestClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Members'
    var_11 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_11 is True
    var_12 = 'int'
    var_13 = bool('int' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc['test_module.EmptyClass'])
    assert var_9 is True
    var_10 = 'Bases'
    var_11 = bool('Bases' not in var_0.doc['test_module.EmptyClass'])
    assert var_11 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_nonexistent_intermediate_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_none_intermediate_value. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'nonexistent.value'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = None
    var_1 = 'inner.value.something'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_class_api_predicate_line_38_true. Retrieved 6/52 statements.


def test_case_0():
    var_0 = 'MEMBER1'
    var_1 = None
    var_2 = []
    var_3 = 1
    var_4 = []
    var_5 = [var_1]
    var_6 = 'test_module'
    var_7 = 'test_module.TestEnum'
    var_8 = 'Enums table'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_api_function_def. Retrieved 7/11 statements.
# Partially parsed test_api_async_function_def. Retrieved 7/11 statements.
# Partially parsed test_api_class_def. Retrieved 7/11 statements.
# Partially parsed test_api_with_decorators. Retrieved 7/11 statements.
# Partially parsed test_api_with_prefix. Retrieved 8/13 statements.
# Partially parsed test_api_link_enabled. Retrieved 7/11 statements.
# Partially parsed test_api_link_disabled. Retrieved 7/11 statements.
# Partially parsed test_api_with_docstring. Retrieved 7/11 statements.
# Partially parsed test_api_nested_class. Retrieved 8/13 statements.
# Partially parsed test_api_sets_level. Retrieved 7/11 statements.
# Partially parsed test_api_sets_root. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "\ndef example_func():\n    '''Example function.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.example_func'
    var_7 = bool(var_6 in var_1.doc)
    assert var_7 is True
    var_8 = 'example_func()'
    var_9 = bool('example_func()' in var_1.doc[var_6])
    assert var_9 is True
    var_10 = '*Full name:* `test_module.example_func`'
    var_11 = bool('*Full name:* `test_module.example_func`' in var_1.doc[var_6])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "\nasync def async_func():\n    '''Async function.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.async_func'
    var_7 = bool(var_6 in var_1.doc)
    assert var_7 is True
    var_8 = 'async async_func()'
    var_9 = bool('async async_func()' in var_1.doc[var_6])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "\nclass ExampleClass:\n    '''Example class.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.ExampleClass'
    var_7 = bool(var_6 in var_1.doc)
    assert var_7 is True
    var_8 = 'class ExampleClass'
    var_9 = bool('class ExampleClass' in var_1.doc[var_6])
    assert var_9 is True
    var_10 = '*Full name:* `test_module.ExampleClass`'
    var_11 = bool('*Full name:* `test_module.ExampleClass`' in var_1.doc[var_6])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "\n@staticmethod\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.decorated_func'
    var_7 = bool(var_6 in var_1.doc)
    assert var_7 is True
    var_8 = 'Decorators'
    var_9 = bool('Decorators' in var_1.doc[var_6])
    assert var_9 is True
    var_10 = '@staticmethod'
    var_11 = bool('@staticmethod' in var_1.doc[var_6])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\nclass OuterClass:\n    def inner_method(self):\n        pass\n'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'OuterClass'
    var_7 = 'test_module.OuterClass.inner_method'
    var_8 = bool(var_7 in var_1.doc)
    assert var_8 is True
    var_9 = 'inner_method()'
    var_10 = bool('inner_method()' in var_1.doc[var_7])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef test_func():\n    pass\n'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.test_func'
    var_7 = '<a id='
    var_8 = bool('<a id=' in var_1.doc[var_6])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1)
    var_3 = '\ndef test_func():\n    pass\n'
    var_4 = 'test_module'
    var_5 = var_2.parse(var_4, var_3)
    var_6 = 'test_module.test_func'
    var_7 = '<a id='
    var_8 = bool('<a id=' not in var_2.doc[var_6])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = "\ndef documented_func():\n    '''This is a docstring.'''\n    pass\n"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.documented_func'
    var_7 = bool(var_6 in var_1.docstring)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\nclass OuterClass:\n    class InnerClass:\n        pass\n'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'OuterClass'
    var_7 = 'test_module.OuterClass.InnerClass'
    var_8 = bool(var_7 in var_1.doc)
    assert var_8 is True
    var_9 = 'class InnerClass'
    var_10 = bool('class InnerClass' in var_1.doc[var_7])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef func_in_module():\n    pass\n'
    var_3 = 'test_module.submodule'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.submodule.func_in_module'
    var_7 = var_1.level[var_6]
    var_8 = bool(var_1.level[var_6] == var_1.level[var_3])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    var_2 = '\ndef func():\n    pass\n'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 0
    var_6 = 'test_module.func'
    var_7 = var_1.root[var_6]
    var_8 = bool(var_1.root[var_6] == var_3)
    assert var_8 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_visit_constant_with_syntax_error. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not a valid python expression @#$'
    var_4 = []



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_1 = module_0.Parser(var_0)
    var_2 = var_1.link
    assert var_2 is False
    var_3 = var_1.b_level
    assert var_3 == 1
    var_4 = var_1.toc
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Parser(b_level=var_0)
    var_2 = var_1.b_level
    assert var_2 == 2
    var_3 = var_1.link
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.toc
    assert var_2 is True
    var_3 = var_1.link
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = module_0.Parser(var_0, toc=var_1)
    var_3 = var_2.toc
    assert var_3 is True
    var_4 = var_2.link
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.b_level
    assert var_5 == 3
    var_6 = var_3.toc
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_is_public_predicate_false. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 5 evaluates to False.'
    var_1 = True
    var_2 = False
    var_3 = module_0.Parser(var_1, var_1, var_2)
    var_4 = 'test_module'
    var_5 = var_3.is_public(var_4)
    assert var_5 is False



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_func_api_has_default_false. Retrieved 10/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = 'default_val'
    var_7 = 'root_module'
    var_8 = 'test_func'
    var_9 = False
    var_10 = 'test_func'
    var_11 = bool('test_func' in var_0.doc)
    assert var_11 is True
    var_12 = var_0.doc['test_func']
    var_13 = bool(var_0.doc['test_func'] != '')
    assert var_13 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_class_api_assign_predicate_false. Retrieved 11/30 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a = b = 1'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.targets
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_4.targets
    var_8 = len(var_7)
    var_9 = 1
    var_10 = var_8 == var_9



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/14 statements.
# Partially parsed test_class_api_empty_class. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 10/13 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str = "default"\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = 'attr1'
    var_15 = bool('attr1' in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass(Base):\n    pass\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Bases'
    var_13 = bool('Bases' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.Color'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.Color'
    var_11 = bool('test_module.Color' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass EmptyClass:\n    pass\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.EmptyClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.EmptyClass'
    var_11 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'public_attr'
    var_13 = bool('public_attr' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = '_private_attr'
    var_15 = bool('_private_attr' not in var_0.doc['test_module.TestClass'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr2\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'attr1'
    var_13 = bool('attr1' in var_0.doc['test_module.TestClass'])
    assert var_13 is True
    var_14 = 'attr2'
    var_15 = bool('attr2' not in var_0.doc['test_module.TestClass'])
    assert var_15 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_globals_predicate_evaluates_to_false. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' not in var_0.alias)
    assert var_9 is True
    var_10 = 'test_module.x'
    var_11 = bool('test_module.x' not in var_0.root)
    assert var_11 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/19 statements.


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
    var_10 = 'test_module'
    var_11 = 'test_module.test_func'
    var_12 = False
    var_13 = 'test_module.test_func'
    var_14 = bool('test_module.test_func' in var_0.doc)
    assert var_14 is True
    var_15 = var_0.doc['test_module.test_func']
    var_16 = bool(var_0.doc['test_module.test_func'] != 'Test function')
    assert var_16 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 6/16 statements.
# Partially parsed test_globals_with_assign_and_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 6/19 statements.
# Partially parsed test_globals_with_all_list. Retrieved 6/19 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 6/17 statements.
# Partially parsed test_globals_ignores_non_name_target. Retrieved 7/25 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_annotassign_without_value. Retrieved 6/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = var_0.alias['test_module.x']
    assert var_8 == '42'
    var_9 = var_0.const['test_module.x']
    assert var_9 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST'
    var_3 = 100
    var_4 = []
    var_5 = None
    var_6 = var_0.alias['test_module.CONST']
    assert var_6 == '100'
    var_7 = var_0.root['test_module.CONST']
    var_8 = bool(var_0.root['test_module.CONST'] == var_1)
    assert var_8 is True
    var_9 = var_0.const['test_module.CONST']
    assert var_9 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'value'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = var_0.alias['test_module.value']
    assert var_6 == '42'
    var_7 = var_0.const['test_module.value']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.func1'
    var_10 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.func2'
    var_12 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'item1'
    var_4 = []
    var_5 = 'item2'
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'test_module.item1'
    var_10 = bool('test_module.item1' in var_0.imp[var_1])
    assert var_10 is True
    var_11 = 'test_module.item2'
    var_12 = bool('test_module.item2' in var_0.imp[var_1])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 5
    var_5 = []
    var_6 = None
    var_7 = 'test_module.a'
    var_8 = bool('test_module.a' not in var_0.alias)
    assert var_8 is True
    var_9 = 'test_module.b'
    var_10 = bool('test_module.b' not in var_0.alias)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = []
    var_6 = 2
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.a'
    var_11 = bool('test_module.a' not in var_0.alias)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'UPPER'
    var_3 = 'text'
    var_4 = []
    var_5 = None
    var_6 = var_0.root['test_module.UPPER']
    var_7 = bool(var_0.root['test_module.UPPER'] == var_1)
    assert var_7 is True
    var_8 = var_0.const['test_module.UPPER']
    assert var_8 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x'
    var_3 = 'int'
    var_4 = []
    var_5 = None
    var_6 = 1
    var_7 = 'test_module.x'
    var_8 = bool('test_module.x' not in var_0.alias)
    assert var_8 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0
    var_6 = 'test_module.ospath'
    var_7 = bool(var_6 in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias[var_6]
    assert var_8 == 'os.path'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test_attr_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_broken_chain_in_middle. Retrieved 2/7 statements.
# Partially parsed test_attr_none_value_in_chain. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_numeric_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_with_list_attribute. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'attr'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.value'

def test_case_0():
    var_0 = 'deep_value'
    var_1 = 'level2.level3.data'

def test_case_0():
    var_0 = 'value'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent'

def test_case_0():
    var_0 = 'nested_value'
    var_1 = 'inner.nonexistent.value'

def test_case_0():
    var_0 = None
    var_1 = 'inner.value'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 42
    var_1 = 'num'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'items'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_globals_const_predicate_false. Retrieved 5/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'CONSTANT'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = None
    var_6 = 'test_module'
    var_7 = var_0.const['test.module.CONSTANT']
    var_8 = bool(var_0.const['test.module.CONSTANT'] != 'int')
    assert var_8 is True
    var_9 = var_0.const['test.module.CONSTANT']
    assert var_9 == 'existing_type'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 14/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'test_module'
    var_8 = 'test_module.test_func'
    var_9 = False
    var_10 = 'test_module.test_func'
    var_11 = bool('test_module.test_func' in var_0.doc)
    assert var_11 is True
    var_12 = var_0.doc[var_8]
    var_13 = len(var_12)
    var_14 = 'Test function\n'
    var_15 = len(var_14)
    var_16 = bool(var_13 > var_15)
    assert var_16 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_assignment. Retrieved 8/14 statements.
# Partially parsed test_globals_with_assignment_and_type_comment. Retrieved 9/15 statements.
# Partially parsed test_globals_with_all_list. Retrieved 8/14 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 8/14 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 10/16 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 10/16 statements.
# Partially parsed test_globals_uppercase_constant_stored. Retrieved 8/14 statements.
# Partially parsed test_globals_ignores_annotated_without_value. Retrieved 10/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int = 5'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.x']
    assert var_10 == '5'
    var_11 = var_0.const['test_module.x']
    assert var_11 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MY_CONST = 42'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.MY_CONST'
    var_9 = bool('test_module.MY_CONST' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.MY_CONST']
    assert var_10 == '42'
    var_11 = 'test_module.MY_CONST'
    var_12 = bool('test_module.MY_CONST' in var_0.root)
    assert var_12 is True
    var_13 = var_0.const['test_module.MY_CONST']
    assert var_13 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "x = 'hello'  # type: str"
    var_4 = True
    var_5 = module_1.parse(var_3, type_comments=var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.globals(var_1, var_7)
    var_9 = 'test_module.x'
    var_10 = bool('test_module.x' in var_0.alias)
    assert var_10 is True
    var_11 = var_0.const['test_module.x']
    assert var_11 == 'str'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['func1', 'func2']"
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.func1'
    var_9 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_9 is True
    var_10 = 'test_module.func2'
    var_11 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ('Class1', 'Class2')"
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.Class1'
    var_9 = bool('test_module.Class1' in var_0.imp[var_1])
    assert var_9 is True
    var_10 = 'test_module.Class2'
    var_11 = bool('test_module.Class2' in var_0.imp[var_1])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a, b = 1, 2'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x = y = 5'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONSTANT = 100'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = 'test_module.CONSTANT'
    var_9 = bool('test_module.CONSTANT' in var_0.root)
    assert var_9 is True
    var_10 = var_0.root['test_module.CONSTANT']
    var_11 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int'
    var_4 = module_1.parse(var_3)
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #67
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_element_with_single_constant. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_element_with_multiple_same_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_element_with_multiple_different_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 2/8 statements.
# Partially parsed test_e_type_multiple_elements_different_types. Retrieved 2/8 statements.
# Partially parsed test_e_type_none_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_empty_sequence_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 2/6 statements.
# Partially parsed test_e_type_mixed_constant_and_non_constant. Retrieved 3/8 statements.
# Partially parsed test_e_type_multiple_elements_with_multiple_constants. Retrieved 4/12 statements.
# Partially parsed test_e_type_float_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_bool_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_none_constants. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'string'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'string'
    var_3 = []

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 'x'
    var_1 = None
    var_2 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []

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
    var_0 = 1.5
    var_1 = []
    var_2 = 2.5
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []

def test_case_0():
    var_0 = None
    var_1 = []



# Parsed testcases at query #68
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_none_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_single_empty_sequence. Retrieved 1/3 statements.
# Partially parsed test_e_type_single_constant_int. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_constant_str. Retrieved 1/5 statements.
# Partially parsed test_e_type_single_constant_float. Retrieved 1/5 statements.
# Partially parsed test_e_type_multiple_same_type_constants. Retrieved 3/9 statements.
# Partially parsed test_e_type_multiple_different_type_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_sequences_same_type. Retrieved 3/10 statements.
# Partially parsed test_e_type_multiple_sequences_different_types. Retrieved 2/8 statements.
# Partially parsed test_e_type_multiple_sequences_mixed_types_in_sequence. Retrieved 3/10 statements.
# Partially parsed test_e_type_sequence_with_none_constant_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

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
    var_0 = 1
    var_1 = []
    var_2 = 'hello'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'hello'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'hello'
    var_3 = []
    var_4 = 42
    var_5 = []

def test_case_0():
    var_0 = None
    var_1 = []



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_func_api_predicate_false. Retrieved 17/28 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 42
    var_8 = []
    var_9 = globals()
    var_10 = 'table'
    var_11 = 'table_result'
    var_12 = lambda *args, **kwargs: var_11
    var_13 = globals()
    var_14 = 'code'
    var_15 = lambda x: f'code({x})'
    var_16 = 'test_module'
    var_17 = 'test_module.test_func'
    var_18 = False
    var_19 = 'test_module.test_func'
    var_20 = bool('test_module.test_func' in var_0.doc)
    assert var_20 is True
    var_21 = var_0.doc['test_module.test_func']
    var_22 = bool(var_0.doc['test_module.test_func'] != '# test_func\n\n')
    assert var_22 is True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_compile_basic. Retrieved 4/9 statements.
# Partially parsed test_compile_with_toc. Retrieved 5/10 statements.
# Partially parsed test_compile_multiple_items. Retrieved 4/13 statements.
# Partially parsed test_compile_with_magic_method. Retrieved 4/12 statements.
# Partially parsed test_compile_with_constants. Retrieved 5/12 statements.
# Partially parsed test_compile_missing_docstring. Retrieved 4/12 statements.
# Partially parsed test_compile_with_link. Retrieved 4/9 statements.
# Partially parsed test_compile_nested_items. Retrieved 5/18 statements.
# Partially parsed test_compile_all_filter. Retrieved 5/14 statements.
# Partially parsed test_compile_sort_order. Retrieved 3/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_3)
    assert var_5 is True
    var_6 = 'Test module docstring'
    var_7 = bool('Test module docstring' in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '**Table of contents:**'
    var_6 = bool('**Table of contents:**' in var_4)
    assert var_6 is True
    var_7 = 'test_module'
    var_8 = bool('test_module' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'pkg'
    var_5 = bool('pkg' in var_3)
    assert var_5 is True
    var_6 = 'func'
    var_7 = bool('func' in var_3)
    assert var_7 is True
    var_8 = 'Package docstring'
    var_9 = bool('Package docstring' in var_3)
    assert var_9 is True
    var_10 = 'Function docstring'
    var_11 = bool('Function docstring' in var_3)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'pkg'
    var_5 = bool('pkg' in var_3)
    assert var_5 is True
    var_6 = '__init__'
    var_7 = bool('__init__' not in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'pkg'
    var_4 = var_2.compile()
    var_5 = 'Constants'
    var_6 = bool('Constants' in var_4)
    assert var_6 is True
    var_7 = 'VERSION'
    var_8 = bool('VERSION' in var_4)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    var_4 = 'pkg'
    var_5 = bool('pkg' in var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.compile()
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_3)
    assert var_5 is True
    var_6 = 'Test module docstring'
    var_7 = bool('Test module docstring' in var_3)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.compile()
    var_5 = '**Table of contents:**'
    var_6 = bool('**Table of contents:**' in var_4)
    assert var_6 is True
    var_7 = 'pkg'
    var_8 = bool('pkg' in var_4)
    assert var_8 is True
    var_9 = 'subpkg'
    var_10 = bool('subpkg' in var_4)
    assert var_10 is True
    var_11 = 'func'
    var_12 = bool('func' in var_4)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'pkg.public_func'
    var_4 = var_2.compile()
    var_5 = 'public_func'
    var_6 = bool('public_func' in var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.compile()
    assert var_3 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_func_ann_with_self_parameter. Retrieved 8/15 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_annotations. Retrieved 8/17 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 8/16 statements.
# Partially parsed test_func_ann_without_self. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_self_annotation. Retrieved 9/18 statements.
# Partially parsed test_func_ann_classmethod_with_self_annotation. Retrieved 7/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'module'
    var_9 = True
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'module'
    var_9 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = None
    var_4 = []
    var_5 = 'return'
    var_6 = 'str'
    var_7 = []
    var_8 = 'module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = '*'
    var_5 = []
    var_6 = 'y'
    var_7 = []
    var_8 = 'return'
    var_9 = []
    var_10 = 'module'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = []
    var_4 = 'b'
    var_5 = []
    var_6 = 'return'
    var_7 = []
    var_8 = 'module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'MyClass'
    var_3 = None
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = 'return'
    var_8 = []
    var_9 = 'module'
    var_10 = True
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'type[MyClass]'
    var_3 = None
    var_4 = []
    var_5 = 'return'
    var_6 = []
    var_7 = 'module'
    var_8 = True



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/14 statements.
# Partially parsed test_class_api_with_members. Retrieved 5/15 statements.
# Partially parsed test_class_api_with_enums. Retrieved 5/15 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 5/15 statements.
# Partially parsed test_class_api_empty_class. Retrieved 5/15 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass(BaseClass):\n    pass\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = 'TestClass'
    var_6 = bool('TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'Bases'
    var_8 = bool('Bases' in var_0.doc['TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    x: int\n    y: str\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = 'TestClass'
    var_6 = bool('TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'Members'
    var_8 = bool('Members' in var_0.doc['TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestEnum(enum.Enum):\n    A = 1\n    B = 2\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestEnum'
    var_5 = 'TestEnum'
    var_6 = bool('TestEnum' in var_0.doc)
    assert var_6 is True
    var_7 = 'Enums'
    var_8 = bool('Enums' in var_0.doc['TestEnum'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    x: int\n    del x\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = 'TestClass'
    var_6 = bool('TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'x'
    var_8 = bool('x' not in var_0.doc['TestClass'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    pass\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = 'TestClass'
    var_6 = bool('TestClass' in var_0.doc)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    x: int\n    _private: str\n'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'TestClass'
    var_5 = 'TestClass'
    var_6 = bool('TestClass' in var_0.doc)
    assert var_6 is True
    var_7 = '_private'
    var_8 = bool('_private' not in var_0.doc['TestClass'])
    assert var_8 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_func_ann_predicate_line_7. Retrieved 6/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'int'
    var_3 = []
    var_4 = 'test_module'
    var_5 = True
    var_6 = False



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_class_api_predicate_line_11_evaluates_to_false. Retrieved 7/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'del x'
    var_2 = module_1.parse(var_1)
    var_3 = var_2.body
    var_4 = 0
    var_5 = var_3[var_4]
    var_6 = var_5.target



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 8/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = None
    var_4 = []
    var_5 = 123
    var_6 = []
    var_7 = 'test_module'
    var_8 = var_1.imp[var_7]
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/10 statements.
# Partially parsed test_imports_simple_import_with_asname. Retrieved 4/10 statements.
# Partially parsed test_imports_multiple_names. Retrieved 6/13 statements.
# Partially parsed test_imports_from_import_absolute. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_with_asname. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_1. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_relative_level_2. Retrieved 6/12 statements.
# Partially parsed test_imports_from_import_multiple_names. Retrieved 8/15 statements.
# Partially parsed test_imports_from_import_none_module. Retrieved 5/11 statements.
# Partially parsed test_imports_star_import. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = var_0.alias['test_module.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = var_0.alias['test_module.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = None
    var_4 = 'sys'
    var_5 = 'system'
    var_6 = var_0.alias['test_module.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['test_module.system']
    assert var_7 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['test_module.path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0
    var_6 = var_0.alias['test_module.ospath']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.test_module'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['pkg.test_module.func']
    assert var_6 == 'pkg.sibling.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.subpkg.test_module'
    var_2 = 'other'
    var_3 = 'func'
    var_4 = None
    var_5 = 2
    var_6 = var_0.alias['pkg.subpkg.test_module.func']
    assert var_6 == 'pkg.other.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 'getcwd'
    var_6 = 'get_cwd'
    var_7 = 0
    var_8 = var_0.alias['test_module.path']
    assert var_8 == 'os.path'
    var_9 = var_0.alias['test_module.get_cwd']
    assert var_9 == 'os.getcwd'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = None
    var_3 = 'func'
    var_4 = 1
    var_5 = 'test_module.func'
    var_6 = bool('test_module.func' not in var_0.alias)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = '*'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['test_module.*']
    assert var_6 == 'os.*'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_class_api_delete_non_name_target. Retrieved 6/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'obj'
    var_2 = []
    var_3 = 'attr'
    var_4 = 'test_module'
    var_5 = 'test_module.TestClass'
    var_6 = []
    var_7 = var_0.doc['test_module.TestClass']
    assert var_7 == '## class TestClass\n\n'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_globals_predicate_line_8_evaluates_to_false. Retrieved 27/45 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False for various cases.'
    var_1 = module_0.Parser()
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = var_1.alias
    var_8 = len(var_7)
    var_9 = 'test_module'
    var_10 = var_1.alias
    var_11 = len(var_10)
    var_12 = var_11 > var_8
    var_13 = var_1.alias
    var_14 = len(var_13)
    var_15 = var_14 == var_8
    var_16 = bool(var_12 or var_15)
    assert var_16 is True
    var_17 = []
    var_18 = []
    var_19 = 'int'
    var_20 = []
    var_21 = 10
    var_22 = []
    var_23 = 1
    var_24 = var_1.alias
    var_25 = len(var_24)
    var_26 = var_1.alias
    var_27 = len(var_26)
    var_28 = bool(var_27 == var_25)
    assert var_28 is True
    var_29 = 'y'
    var_30 = []
    var_31 = []
    var_32 = var_1.alias
    var_33 = len(var_32)
    var_34 = var_1.alias
    var_35 = len(var_34)
    var_36 = bool(var_35 == var_33)
    assert var_36 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_func_api_simple_function. Retrieved 11/22 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 10/21 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/22 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/21 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 11/21 statements.
# Partially parsed test_func_api_classmethod. Retrieved 11/21 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 10/21 statements.


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
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '| x | y | return |'
    var_16 = bool('| x | y | return |' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 5
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '| x | return |'
    var_15 = bool('| x | return |' in var_0.doc['test_module.func'])
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
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True
    var_13 = False
    var_14 = 'test_module.MyClass.method'
    var_15 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_15 is True
    var_16 = 'Self'
    var_17 = bool('Self' in var_0.doc['test_module.MyClass.method'])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'args'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '*args'
    var_16 = bool('*args' in var_0.doc['test_module.func'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'kwargs'
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = 'x'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.func'
    var_12 = False
    var_13 = 'test_module.func'
    var_14 = bool('test_module.func' in var_0.doc)
    assert var_14 is True
    var_15 = '**kwargs'
    var_16 = bool('**kwargs' in var_0.doc['test_module.func'])
    assert var_16 is True

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
    var_10 = 'test_module'
    var_11 = 'test_module.MyClass.method'
    var_12 = True
    var_13 = 'test_module.MyClass.method'
    var_14 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_14 is True
    var_15 = 'type[Self]'
    var_16 = bool('type[Self]' in var_0.doc['test_module.MyClass.method'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 'y'
    var_6 = []
    var_7 = [var_3]
    var_8 = []
    var_9 = 'test_module'
    var_10 = 'test_module.func'
    var_11 = False
    var_12 = 'test_module.func'
    var_13 = bool('test_module.func' in var_0.doc)
    assert var_13 is True
    var_14 = '| x | * | y | return |'
    var_15 = bool('| x | * | y | return |' in var_0.doc['test_module.func'])
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_walk_body_simple_statements. Retrieved 10/12 statements.
# Partially parsed test_walk_body_if_statement. Retrieved 10/12 statements.
# Partially parsed test_walk_body_nested_if. Retrieved 10/12 statements.
# Partially parsed test_walk_body_try_except. Retrieved 10/12 statements.
# Partially parsed test_walk_body_try_finally. Retrieved 10/12 statements.
# Partially parsed test_walk_body_try_else. Retrieved 12/15 statements.
# Partially parsed test_walk_body_multiple_except_handlers. Retrieved 12/15 statements.
# Partially parsed test_walk_body_pass_statement. Retrieved 8/9 statements.
# Partially parsed test_walk_body_complex_nested. Retrieved 12/15 statements.


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

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    x = 1\nelse:\n    y = 2'
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

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    if False:\n        x = 1\n    y = 2'
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

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    y = 2'
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

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nfinally:\n    y = 2'
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

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept:\n    pass\nelse:\n    y = 2'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept KeyError:\n    z = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]

import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'pass'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\nz = 3'
    var_1 = module_0.parse(var_0)
    var_2 = var_1.body
    var_3 = module_1.walk_body(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_globals_with_annassign. Retrieved 4/12 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/12 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 5/14 statements.
# Partially parsed test_globals_with_all_list. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 4/12 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/13 statements.
# Partially parsed test_globals_ignores_lowercase_non_all. Retrieved 4/12 statements.
# Partially parsed test_globals_annassign_without_value. Retrieved 4/12 statements.
# Partially parsed test_globals_multiple_targets. Retrieved 4/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int = 5'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.x']
    assert var_6 == '5'
    var_7 = var_0.const['test_module.x']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Y = 10'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Y'
    var_5 = bool('test_module.Y' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.Y']
    assert var_6 == '10'
    var_7 = 'test_module.Y'
    var_8 = bool('test_module.Y' in var_0.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ('func1', 'func2')"
    var_4 = 0
    var_5 = 'test_module.func1'
    var_6 = bool('test_module.func1' in var_0.imp['test_module'])
    assert var_6 is True
    var_7 = 'test_module.func2'
    var_8 = bool('test_module.func2' in var_0.imp['test_module'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = "__all__ = ['item1', 'item2']"
    var_4 = 0
    var_5 = 'test_module.item1'
    var_6 = bool('test_module.item1' in var_0.imp['test_module'])
    assert var_6 is True
    var_7 = 'test_module.item2'
    var_8 = bool('test_module.item2' in var_0.imp['test_module'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a, b = 1, 2'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.a'
    var_5 = bool('test_module.a' not in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.b'
    var_7 = bool('test_module.b' not in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5  # type: int'
    var_2 = True
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.const)
    assert var_6 is True
    var_7 = var_0.const['test_module.x']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 5'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.x'
    var_7 = bool('test_module.x' not in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x: int'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = y = 5'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_enum. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 6/13 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 7/14 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 6/13 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    x: int\n    y: str = "default"\n    '
    var_3 = 0
    var_4 = 'test_module.MyClass'
    var_5 = []
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' in var_0.doc['test_module.MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Parent:\n    pass\n\nclass Child(Parent):\n    pass\n    '
    var_3 = 1
    var_4 = 'test_module.Child'
    var_5 = 'test_module.Child'
    var_6 = bool('test_module.Child' in var_0.doc)
    assert var_6 is True
    var_7 = 'Bases'
    var_8 = bool('Bases' in var_0.doc['test_module.Child'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass Color:\n    RED: str\n    GREEN: str\n    BLUE: str\n    '
    var_3 = 0
    var_4 = 'test_module.Color'
    var_5 = []
    var_6 = 'test_module.Color'
    var_7 = bool('test_module.Color' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    x: int\n    y: str\n    del y\n    '
    var_3 = 0
    var_4 = 'test_module.MyClass'
    var_5 = []
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    x = 42  # type: int\n    '
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.MyClass'
    var_6 = []
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_0.doc)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    public: int\n    _private: str\n    '
    var_3 = 0
    var_4 = 'test_module.MyClass'
    var_5 = []
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nclass EmptyClass:\n    pass\n    '
    var_3 = 0
    var_4 = 'test_module.EmptyClass'
    var_5 = []
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.path.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'print'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os._path'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private.module'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'os.__all__.join'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__main__.__dict__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_globals_with_annotated_assign. Retrieved 4/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/13 statements.
# Partially parsed test_globals_with_uppercase_constant. Retrieved 4/13 statements.
# Partially parsed test_globals_with_all_list. Retrieved 4/13 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 4/13 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/14 statements.
# Partially parsed test_globals_ignores_non_name_targets. Retrieved 4/13 statements.
# Partially parsed test_globals_ignores_multiple_targets. Retrieved 4/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 5'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.x']
    assert var_6 == '5'
    var_7 = var_0.const['test_module.x']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'y = 10'
    var_3 = 0
    var_4 = 'test_module.y'
    var_5 = bool('test_module.y' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.y']
    assert var_6 == '10'
    var_7 = var_0.const['test_module.y']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONSTANT = 42'
    var_3 = 0
    var_4 = 'test_module.CONSTANT'
    var_5 = bool('test_module.CONSTANT' in var_0.root)
    assert var_5 is True
    var_6 = var_0.root['test_module.CONSTANT']
    var_7 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_7 is True
    var_8 = var_0.const['test_module.CONSTANT']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ['func1', 'func2']"
    var_3 = 0
    var_4 = 'test_module.func1'
    var_5 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_5 is True
    var_6 = 'test_module.func2'
    var_7 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ('item1', 'item2')"
    var_3 = 0
    var_4 = 'test_module.item1'
    var_5 = bool('test_module.item1' in var_0.imp[var_1])
    assert var_5 is True
    var_6 = 'test_module.item2'
    var_7 = bool('test_module.item2' in var_0.imp[var_1])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "z = 'hello'  # type: str"
    var_3 = 0
    var_4 = True
    var_5 = 'test_module.z'
    var_6 = bool('test_module.z' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.const['test_module.z']
    assert var_7 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'a, b = 1, 2'
    var_3 = 0
    var_4 = 'test_module.a'
    var_5 = bool('test_module.a' not in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.b'
    var_7 = bool('test_module.b' not in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = y = 5'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 7/9 statements.
# Partially parsed test_imports_import_with_alias. Retrieved 7/9 statements.
# Partially parsed test_imports_multiple_imports. Retrieved 7/9 statements.
# Partially parsed test_imports_from_import. Retrieved 7/9 statements.
# Partially parsed test_imports_from_import_with_alias. Retrieved 7/9 statements.
# Partially parsed test_imports_from_import_multiple. Retrieved 7/9 statements.
# Partially parsed test_imports_relative_import_level_1. Retrieved 7/9 statements.
# Partially parsed test_imports_relative_import_level_2. Retrieved 7/9 statements.
# Partially parsed test_imports_relative_import_with_module. Retrieved 7/9 statements.
# Partially parsed test_imports_from_import_star. Retrieved 7/9 statements.
# Partially parsed test_imports_nested_package. Retrieved 7/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'import os'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.os']
    assert var_7 == 'os'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'import os as operating_system'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.operating_system']
    assert var_7 == 'os'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'import os, sys'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.os']
    assert var_7 == 'os'
    var_8 = var_0.alias['mymodule.sys']
    assert var_8 == 'sys'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'from os import path'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.path']
    assert var_7 == 'os.path'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'from os import path as p'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.p']
    assert var_7 == 'os.path'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'from os import path, environ'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.path']
    assert var_7 == 'os.path'
    var_8 = var_0.alias['mymodule.environ']
    assert var_8 == 'os.environ'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = 'from . import sibling'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['package.submodule.sibling']
    assert var_7 == 'package.sibling'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.sub.module'
    var_2 = 'from .. import other'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['package.sub.module.other']
    assert var_7 == 'package.other'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'package.submodule'
    var_2 = 'from .sibling import func'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['package.submodule.func']
    assert var_7 == 'package.sibling.func'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mymodule'
    var_2 = 'from os import *'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['mymodule.*']
    assert var_7 == 'os.*'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = 'import collections.abc'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.imports(var_1, var_5)
    var_7 = var_0.alias['pkg.mod.collections.abc']
    assert var_7 == 'collections.abc'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_class_api_predicate_line_38_true. Retrieved 13/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'enum.Enum'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = [var_7]
    var_9 = '\nRED = 1\nGREEN = 2\nBLUE = 3\n'
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body
    var_12 = var_0.class_api(var_1, var_2, var_8, var_11)
    var_13 = 'Enums'
    var_14 = bool('Enums' in var_0.doc[var_2])
    assert var_14 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_class_api_delete_statement_predicate. Retrieved 4/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = None
    var_3 = []
    var_4 = 'attr2'
    var_5 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parse_basic_module. Retrieved 6/7 statements.
# Partially parsed test_parse_with_imports. Retrieved 6/7 statements.
# Partially parsed test_parse_with_higher_base_level. Retrieved 8/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with a basic module.'
    var_1 = module_0.Parser()
    var_2 = 'x = 1\ndef foo(): pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module'
    var_6 = bool('test_module' in var_1.doc)
    assert var_6 is True
    var_7 = 'test_module'
    var_8 = bool('test_module' in var_1.level)
    assert var_8 is True
    var_9 = 'test_module'
    var_10 = bool('test_module' in var_1.imp)
    assert var_10 is True
    var_11 = 'test_module'
    var_12 = bool('test_module' in var_1.root)
    assert var_12 is True
    var_13 = var_1.level['test_module']
    assert var_13 == 0
    var_14 = var_1.root['test_module']
    assert var_14 == 'test_module'
    var_15 = var_1.imp[var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with import statements.'
    var_1 = module_0.Parser()
    var_2 = 'import os\nfrom sys import path'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module'
    var_6 = bool('test_module' in var_1.alias)
    assert var_6 is True
    var_7 = var_1.imp[var_3]

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with module docstring.'
    var_1 = module_0.Parser()
    var_2 = '"""Module docstring."""\ndef foo(): pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module'
    var_6 = bool('test_module' in var_1.docstring)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with function definition.'
    var_1 = module_0.Parser()
    var_2 = 'def foo():\n    pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.foo'
    var_6 = bool('test_module.foo' in var_1.doc)
    assert var_6 is True
    var_7 = 'test_module.foo'
    var_8 = bool('test_module.foo' in var_1.level)
    assert var_8 is True
    var_9 = 'test_module.foo'
    var_10 = bool('test_module.foo' in var_1.root)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with class definition.'
    var_1 = module_0.Parser()
    var_2 = 'class MyClass:\n    pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.MyClass'
    var_6 = bool('test_module.MyClass' in var_1.doc)
    assert var_6 is True
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_1.level)
    assert var_8 is True
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_1.root)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with nested class methods.'
    var_1 = module_0.Parser()
    var_2 = 'class MyClass:\n    def method(self): pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.MyClass'
    var_6 = bool('test_module.MyClass' in var_1.doc)
    assert var_6 is True
    var_7 = 'test_module.MyClass.method'
    var_8 = bool('test_module.MyClass.method' in var_1.doc)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with constant assignments.'
    var_1 = module_0.Parser()
    var_2 = "CONSTANT = 42\nVARIABLE = 'test'"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.CONSTANT'
    var_6 = bool('test_module.CONSTANT' in var_1.const)
    assert var_6 is True
    var_7 = 'test_module.VARIABLE'
    var_8 = bool('test_module.VARIABLE' in var_1.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with nested module path.'
    var_1 = module_0.Parser()
    var_2 = 'def foo(): pass'
    var_3 = 'package.submodule'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = var_1.level['package.submodule']
    assert var_5 == 1
    var_6 = var_1.root['package.submodule']
    assert var_6 == 'package.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with link option enabled.'
    var_1 = True
    var_2 = module_0.Parser(var_1)
    var_3 = 'def foo(): pass'
    var_4 = 'test_module'
    var_5 = var_2.parse(var_4, var_3)
    var_6 = '<a id='
    var_7 = bool('<a id=' in var_2.doc['test_module'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with async function.'
    var_1 = module_0.Parser()
    var_2 = 'async def async_foo():\n    pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.async_foo'
    var_6 = bool('test_module.async_foo' in var_1.doc)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with type-annotated variables.'
    var_1 = module_0.Parser()
    var_2 = "x: int = 1\ny: str = 'test'"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_1.const)
    assert var_6 is True
    var_7 = 'test_module.y'
    var_8 = bool('test_module.y' in var_1.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with __all__ definition.'
    var_1 = module_0.Parser()
    var_2 = "__all__ = ['foo', 'bar']\ndef foo(): pass\ndef bar(): pass"
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = 'test_module.foo'
    var_6 = bool('test_module.foo' in var_1.imp['test_module'])
    assert var_6 is True
    var_7 = 'test_module.bar'
    var_8 = bool('test_module.bar' in var_1.imp['test_module'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method called multiple times.'
    var_1 = module_0.Parser()
    var_2 = 'module1'
    var_3 = 'def foo(): pass'
    var_4 = var_1.parse(var_2, var_3)
    var_5 = 'module2'
    var_6 = 'def bar(): pass'
    var_7 = var_1.parse(var_5, var_6)
    var_8 = 'module1'
    var_9 = bool('module1' in var_1.doc)
    assert var_9 is True
    var_10 = 'module2'
    var_11 = bool('module2' in var_1.doc)
    assert var_11 is True
    var_12 = 'module1.foo'
    var_13 = bool('module1.foo' in var_1.doc)
    assert var_13 is True
    var_14 = 'module2.bar'
    var_15 = bool('module2.bar' in var_1.doc)
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with different base level.'
    var_1 = 2
    var_2 = module_0.Parser(b_level=var_1)
    var_3 = 'def foo(): pass'
    var_4 = 'test_module'
    var_5 = var_2.parse(var_4, var_3)
    var_6 = var_2.doc[var_4]
    var_7 = '###'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test parse method with table of contents enabled.'
    var_1 = True
    var_2 = module_0.Parser(toc=var_1)
    var_3 = 'def foo(): pass'
    var_4 = 'test_module'
    var_5 = var_2.parse(var_4, var_3)
    var_6 = var_2.link
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = '42'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = "'hello'"
    var_6 = module_0.parse(var_5)
    var_7 = var_6.body[var_0]
    var_8 = var_7.value
    var_9 = [var_4, var_8]
    var_10 = module_1._defaults(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = '`42`'
    var_14 = bool('`42`' in var_11[0])
    assert var_14 is True
    var_15 = "`'hello'`"
    var_16 = bool("`'hello'`" in var_11[1])
    assert var_16 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = 'True'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = None
    var_6 = [var_5, var_4, var_5]
    var_7 = module_1._defaults(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8[0]
    assert var_10 == ' '
    var_11 = var_8[2]
    assert var_11 == ' '

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = "'a|b'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = '&#124;'
    var_10 = bool('&#124;' in var_7[0])
    assert var_10 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = "'a&b'"
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = [var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = '<code>'
    var_10 = bool('<code>' in var_7[0])
    assert var_10 is True
    var_11 = '</code>'
    var_12 = bool('</code>' in var_7[0])
    assert var_12 is True

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
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' '])
    assert var_4 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 0
    var_1 = '1'
    var_2 = module_0.parse(var_1)
    var_3 = var_2.body[var_0]
    var_4 = var_3.value
    var_5 = '2'
    var_6 = module_0.parse(var_5)
    var_7 = var_6.body[var_0]
    var_8 = var_7.value
    var_9 = '3'
    var_10 = module_0.parse(var_9)
    var_11 = var_10.body[var_0]
    var_12 = var_11.value
    var_13 = [var_4, var_8, var_12]
    var_14 = module_1._defaults(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 3



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_element_with_int_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_element_with_str_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_element_with_float_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_multiple_elements_with_int_constants. Retrieved 2/8 statements.
# Partially parsed test_e_type_multiple_elements_mixed_types. Retrieved 3/10 statements.
# Partially parsed test_e_type_none_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_empty_sequence_element. Retrieved 1/3 statements.
# Partially parsed test_e_type_non_constant_element. Retrieved 1/5 statements.
# Partially parsed test_e_type_mixed_constant_and_non_constant. Retrieved 2/7 statements.
# Partially parsed test_e_type_single_constant_in_element. Retrieved 1/5 statements.
# Partially parsed test_e_type_bool_constants. Retrieved 2/7 statements.
# Partially parsed test_e_type_none_constants. Retrieved 1/6 statements.
# Partially parsed test_e_type_three_elements_same_type. Retrieved 3/11 statements.
# Partially parsed test_e_type_three_elements_mixed_types. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []

def test_case_0():
    var_0 = 1.5
    var_1 = []
    var_2 = 2.5
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []
    var_4 = 2
    var_5 = []

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = []
    var_1 = [var_0]

def test_case_0():
    var_0 = 'x'
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 42
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []
    var_4 = 2.5
    var_5 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_api_function_def. Retrieved 7/11 statements.
# Partially parsed test_api_async_function_def. Retrieved 7/11 statements.
# Partially parsed test_api_class_def. Retrieved 7/11 statements.
# Partially parsed test_api_with_decorators. Retrieved 7/12 statements.
# Partially parsed test_api_with_prefix. Retrieved 8/12 statements.
# Partially parsed test_api_full_name_format. Retrieved 7/11 statements.
# Partially parsed test_api_with_link. Retrieved 8/12 statements.
# Partially parsed test_api_nested_class_methods. Retrieved 10/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def my_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.my_func'
    var_8 = bool('test_module.my_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'my_func()'
    var_10 = bool('my_func()' in var_0.doc['test_module.my_func'])
    assert var_10 is True
    var_11 = var_0.root['test_module.my_func']
    assert var_11 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.async_func'
    var_8 = bool('test_module.async_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'async async_func()'
    var_10 = bool('async async_func()' in var_0.doc['test_module.async_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.MyClass'
    var_8 = bool('test_module.MyClass' in var_0.doc)
    assert var_8 is True
    var_9 = 'class MyClass'
    var_10 = bool('class MyClass' in var_0.doc['test_module.MyClass'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\ndef decorated_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'test_module.decorated_func'
    var_8 = bool('test_module.decorated_func' in var_0.doc)
    assert var_8 is True
    var_9 = 'Decorators'
    var_10 = bool('Decorators' in var_0.doc['test_module.decorated_func'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def inner_func(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'OuterClass'
    var_7 = var_0.api(var_5, var_4, prefix=var_6)
    var_8 = 'test_module.OuterClass.inner_func'
    var_9 = bool('test_module.OuterClass.inner_func' in var_0.doc)
    assert var_9 is True
    var_10 = var_0.level['test_module.OuterClass.inner_func']
    assert var_10 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func_with_underscore(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = var_0.api(var_5, var_4)
    var_7 = '*Full name:* `test_module.func_with_underscore`'
    var_8 = bool('*Full name:* `test_module.func_with_underscore`' in var_0.doc['test_module.func_with_underscore'])
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'def my_func(): pass'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = var_1.api(var_6, var_5)
    var_8 = '<a id='
    var_9 = bool('<a id=' in var_1.doc['test_module.my_func'])
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class OuterClass:\n    def method(self): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.body[var_3]
    var_6 = 'test_module'
    var_7 = var_0.api(var_6, var_4)
    var_8 = 'OuterClass'
    var_9 = var_0.api(var_6, var_5, prefix=var_8)
    var_10 = 'test_module.OuterClass'
    var_11 = bool('test_module.OuterClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'test_module.OuterClass.method'
    var_13 = bool('test_module.OuterClass.method' in var_0.doc)
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_globals_predicate_line_18_false. Retrieved 12/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = []
    var_4 = 'y'
    var_5 = []
    var_6 = 5
    var_7 = []
    var_8 = 'test_module'
    var_9 = var_0.alias
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = var_0.const
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = var_0.root
    var_14 = len(var_13)
    assert var_14 == 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_globals_with_annotated_assignment. Retrieved 7/19 statements.
# Partially parsed test_globals_with_annotated_assignment_uppercase. Retrieved 7/19 statements.
# Partially parsed test_globals_with_simple_assignment. Retrieved 6/17 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/17 statements.
# Partially parsed test_globals_with_all_list. Retrieved 7/22 statements.
# Partially parsed test_globals_with_all_tuple. Retrieved 7/22 statements.
# Partially parsed test_globals_with_multiple_targets. Retrieved 7/20 statements.
# Partially parsed test_globals_with_invalid_annotation_target. Retrieved 9/24 statements.
# Partially parsed test_globals_uppercase_constant_type_inference. Retrieved 6/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x'
    var_4 = 'int'
    var_5 = []
    var_6 = 42
    var_7 = []
    var_8 = 1
    var_9 = 'test_module.x'
    var_10 = bool('test_module.x' in var_0.alias)
    assert var_10 is True
    var_11 = var_0.alias['test_module.x']
    assert var_11 == '42'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONSTANT'
    var_4 = 'int'
    var_5 = []
    var_6 = 100
    var_7 = []
    var_8 = 1
    var_9 = 'test_module.CONSTANT'
    var_10 = bool('test_module.CONSTANT' in var_0.root)
    assert var_10 is True
    var_11 = var_0.root['test_module.CONSTANT']
    var_12 = bool(var_0.root['test_module.CONSTANT'] == var_1)
    assert var_12 is True
    var_13 = 'test_module.CONSTANT'
    var_14 = bool('test_module.CONSTANT' in var_0.const)
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'y'
    var_4 = 'hello'
    var_5 = []
    var_6 = None
    var_7 = 'test_module.y'
    var_8 = bool('test_module.y' in var_0.alias)
    assert var_8 is True
    var_9 = var_0.alias['test_module.y']
    assert var_9 == "'hello'"

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'z'
    var_4 = 3.14
    var_5 = []
    var_6 = 'float'
    var_7 = 'test_module.z'
    var_8 = bool('test_module.z' in var_0.const)
    assert var_8 is True
    var_9 = var_0.const['test_module.z']
    assert var_9 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'func1'
    var_5 = []
    var_6 = 'func2'
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.func1'
    var_11 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_11 is True
    var_12 = 'test_module.func2'
    var_13 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = '__all__'
    var_4 = 'ClassA'
    var_5 = []
    var_6 = 'ClassB'
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = 'test_module.ClassA'
    var_11 = bool('test_module.ClassA' in var_0.imp[var_1])
    assert var_11 is True
    var_12 = 'test_module.ClassB'
    var_13 = bool('test_module.ClassB' in var_0.imp[var_1])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 5
    var_6 = []
    var_7 = None
    var_8 = 'test_module.a'
    var_9 = bool('test_module.a' not in var_0.alias)
    assert var_9 is True
    var_10 = 'test_module.b'
    var_11 = bool('test_module.b' not in var_0.alias)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x'
    var_4 = 'int'
    var_5 = []
    var_6 = 10
    var_7 = []
    var_8 = 1
    var_9 = var_0.alias
    var_10 = len(var_9)
    assert var_10 == 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'MAX_VALUE'
    var_4 = 999
    var_5 = []
    var_6 = None
    var_7 = var_0.const['test_module.MAX_VALUE']
    assert var_7 == 'int'
    var_8 = var_0.root['test_module.MAX_VALUE']
    var_9 = bool(var_0.root['test_module.MAX_VALUE'] == var_1)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_docstring. Retrieved 4/26 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = var_2.docstring['test_module']
    assert var_4 == 'This is the module docstring.'
    var_5 = var_2.docstring['test_module.func']
    assert var_5 == 'This is the function docstring.'
    var_6 = var_2.docstring['test_module.MyClass']
    assert var_6 == 'This is the class docstring.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_api_predicate_line_1. Retrieved 19/27 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_func'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test_root'
    var_12 = ''
    var_13 = '#'
    var_14 = var_0.b_level
    var_15 = 2
    var_16 = var_14 + var_15
    var_17 = var_13 * var_16
    assert var_17 == '###'
    var_18 = 'test_root.test_func'
    var_19 = bool('test_root.test_func' in var_0.doc)
    assert var_19 is True
    var_20 = 'test_root.test_func'
    var_21 = var_0.doc[var_20]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_class_api_mem_predicate_evaluates_to_true. Retrieved 9/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = None
    var_6 = []
    var_7 = 'str'
    var_8 = []
    var_9 = 'test'
    var_10 = []
    var_11 = 1
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc[var_2])
    assert var_13 is True
    var_14 = 'Type'
    var_15 = bool('Type' in var_0.doc[var_2])
    assert var_15 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_func_api_vararg_is_not_none. Retrieved 11/17 statements.


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
    var_9 = 'test_module'
    var_10 = 'test_module.test_func'
    var_11 = False
    var_12 = 'test_module.test_func'
    var_13 = bool('test_module.test_func' in var_0.doc)
    assert var_13 is True
    var_14 = '*args'
    var_15 = bool('*args' in var_0.doc['test_module.test_func'])
    assert var_15 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_imports_updates_imp_set. Retrieved 5/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.os'
    var_5 = bool('test_module.os' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.os']
    assert var_6 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os as operating_system'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.operating_system'
    var_5 = bool('test_module.operating_system' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.operating_system']
    assert var_6 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os, sys'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.os'
    var_5 = bool('test_module.os' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.sys'
    var_7 = bool('test_module.sys' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.os']
    assert var_8 == 'os'
    var_9 = var_0.alias['test_module.sys']
    assert var_9 == 'sys'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.path'
    var_5 = bool('test_module.path' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path as file_path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.file_path'
    var_5 = bool('test_module.file_path' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.file_path']
    assert var_6 == 'os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from . import submodule'
    var_2 = 'test_module.sub'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.sub.submodule'
    var_5 = bool('test_module.sub.submodule' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.sub.submodule']
    assert var_6 == 'test_module.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from .. import sibling'
    var_2 = 'test_module.sub.deep'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.sub.deep.sibling'
    var_5 = bool('test_module.sub.deep.sibling' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.sub.deep.sibling']
    assert var_6 == 'test_module.sibling'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from os import path, getcwd'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.path'
    var_5 = bool('test_module.path' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.getcwd'
    var_7 = bool('test_module.getcwd' in var_0.alias)
    assert var_7 is True
    var_8 = var_0.alias['test_module.path']
    assert var_8 == 'os.path'
    var_9 = var_0.alias['test_module.getcwd']
    assert var_9 == 'os.getcwd'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'from ..parent import func'
    var_2 = 'test_module.child.deep'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.child.deep.func'
    var_5 = bool('test_module.child.deep.func' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.child.deep.func']
    assert var_6 == 'test_module.parent.func'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.imp)
    assert var_5 is True
    var_6 = var_0.imp[var_2]

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.alias
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_func_ann_with_self. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_classmethod. Retrieved 5/11 statements.
# Partially parsed test_func_ann_with_star_separator. Retrieved 6/13 statements.
# Partially parsed test_func_ann_without_self. Retrieved 6/13 statements.
# Partially parsed test_func_ann_with_annotation. Retrieved 5/13 statements.
# Partially parsed test_func_ann_multiple_args. Retrieved 8/19 statements.
# Partially parsed test_func_ann_self_with_annotation. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'root'
    var_5 = True
    var_6 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = 'root'
    var_4 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = 'x'
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'int'
    var_2 = 'x'
    var_3 = 'root'
    var_4 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'root'
    var_6 = False
    var_7 = 'any'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MyClass'
    var_2 = 'self'
    var_3 = 'x'
    var_4 = None
    var_5 = 'root'
    var_6 = True
    var_7 = False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_compile_basic. Retrieved 7/13 statements.
# Partially parsed test_compile_with_toc. Retrieved 8/14 statements.
# Partially parsed test_compile_multiple_names. Retrieved 10/16 statements.
# Partially parsed test_compile_magic_method_skipped. Retrieved 9/15 statements.
# Partially parsed test_compile_private_name_excluded. Retrieved 9/15 statements.
# Partially parsed test_compile_with_constants. Retrieved 9/15 statements.
# Partially parsed test_compile_nested_hierarchy. Retrieved 13/19 statements.
# Partially parsed test_compile_empty_doc. Retrieved 2/8 statements.
# Partially parsed test_compile_with_link_formatting. Retrieved 8/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = '# Module `module`\n\n'
    var_3 = 'Module docstring'
    var_4 = 0
    var_5 = set()
    var_6 = var_0.compile()
    var_7 = '# Module `module`'
    var_8 = bool('# Module `module`' in var_6)
    assert var_8 is True
    var_9 = 'Module docstring'
    var_10 = bool('Module docstring' in var_6)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'module'
    var_3 = '# Module `module`\n\n'
    var_4 = 'Module docstring'
    var_5 = 0
    var_6 = set()
    var_7 = var_1.compile()
    var_8 = '**Table of contents:**'
    var_9 = bool('**Table of contents:**' in var_7)
    assert var_9 is True
    var_10 = '# Module `module`'
    var_11 = bool('# Module `module`' in var_7)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.func'
    var_3 = '# Module `module`\n\n'
    var_4 = '## func()\n\n*Full name:* `module.func`\n\n'
    var_5 = 'Module doc'
    var_6 = 'Function doc'
    var_7 = 0
    var_8 = set()
    var_9 = var_0.compile()
    var_10 = 'Module doc'
    var_11 = bool('Module doc' in var_9)
    assert var_11 is True
    var_12 = 'Function doc'
    var_13 = bool('Function doc' in var_9)
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.__init__'
    var_3 = '# Module `module`\n\n'
    var_4 = '## __init__()\n\n'
    var_5 = 'Module doc'
    var_6 = 0
    var_7 = set()
    var_8 = var_0.compile()
    var_9 = 'Module doc'
    var_10 = bool('Module doc' in var_8)
    assert var_10 is True
    var_11 = '__init__'
    var_12 = bool('__init__' not in var_8)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module._private'
    var_3 = '# Module `module`\n\n'
    var_4 = '## _private\n\n'
    var_5 = 'Module doc'
    var_6 = 0
    var_7 = set()
    var_8 = var_0.compile()
    var_9 = '_private'
    var_10 = bool('_private' not in var_8)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = '# Module `module`\n\n'
    var_3 = 'Module doc'
    var_4 = 'module.CONST'
    var_5 = 0
    var_6 = {var_4}
    var_7 = 'int'
    var_8 = var_0.compile()
    var_9 = 'Constants'
    var_10 = bool('Constants' in var_8)
    assert var_10 is True
    var_11 = 'CONST'
    var_12 = bool('CONST' in var_8)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.cls'
    var_3 = 'module.cls.method'
    var_4 = '# Module `module`\n\n'
    var_5 = '## class cls\n\n*Full name:* `module.cls`\n\n'
    var_6 = '### method()\n\n*Full name:* `module.cls.method`\n\n'
    var_7 = 'Module'
    var_8 = 'Class'
    var_9 = 'Method'
    var_10 = 0
    var_11 = set()
    var_12 = var_0.compile()
    var_13 = 'Module'
    var_14 = bool('Module' in var_12)
    assert var_14 is True
    var_15 = 'Class'
    var_16 = bool('Class' in var_12)
    assert var_16 is True
    var_17 = 'Method'
    var_18 = bool('Method' in var_12)
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = var_0.compile()
    assert var_1 == '\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'module'
    var_3 = '# Module `{}`\n<a id="{}"></a>\n\n'
    var_4 = 'Module doc'
    var_5 = 0
    var_6 = set()
    var_7 = var_1.compile()
    var_8 = 'module'
    var_9 = bool('module' in var_7)
    assert var_9 is True
    var_10 = bool('module-doc' not in var_7 or '#module' in var_7)
    assert var_10 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_globals_predicate_line_33_false. Retrieved 8/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 33 evaluates to False when const already has a value.'
    var_1 = module_0.Parser()
    var_2 = 'module.CONSTANT'
    var_3 = 'int'
    var_4 = 'CONSTANT'
    var_5 = None
    var_6 = []
    var_7 = 42
    var_8 = []
    var_9 = 'module'
    var_10 = var_1.const['module.CONSTANT']
    assert var_10 == 'int'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_api_function_def. Retrieved 4/12 statements.
# Partially parsed test_api_async_function_def. Retrieved 4/12 statements.
# Partially parsed test_api_class_def. Retrieved 4/12 statements.
# Partially parsed test_api_with_decorators. Retrieved 4/12 statements.
# Partially parsed test_api_nested_class_method. Retrieved 5/13 statements.
# Partially parsed test_api_with_link. Retrieved 5/13 statements.
# Partially parsed test_api_underscore_name. Retrieved 4/12 statements.
# Partially parsed test_api_with_docstring. Retrieved 4/13 statements.
# Partially parsed test_api_classmethod. Retrieved 4/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def sample_func(): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.sample_func'
    var_5 = bool('test_module.sample_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'sample_func()'
    var_7 = bool('sample_func()' in var_0.doc['test_module.sample_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def async_func(): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.async_func'
    var_5 = bool('test_module.async_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'async async_func()'
    var_7 = bool('async async_func()' in var_0.doc['test_module.async_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class SampleClass: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.SampleClass'
    var_5 = bool('test_module.SampleClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'class SampleClass'
    var_7 = bool('class SampleClass' in var_0.doc['test_module.SampleClass'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@staticmethod\ndef decorated_func(): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.decorated_func'
    var_5 = bool('test_module.decorated_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'Decorators'
    var_7 = bool('Decorators' in var_0.doc['test_module.decorated_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class OuterClass:\n    def method(self): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = ''
    var_5 = 'test_module.OuterClass'
    var_6 = bool('test_module.OuterClass' in var_0.doc)
    assert var_6 is True
    var_7 = 'test_module.OuterClass.method'
    var_8 = bool('test_module.OuterClass.method' in var_0.doc)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'def func_with_link(): pass'
    var_3 = 0
    var_4 = 'test_module'
    var_5 = '<a id='
    var_6 = bool('<a id=' in var_1.doc['test_module.func_with_link'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func_with_underscores(): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func_with_underscores'
    var_5 = bool('test_module.func_with_underscores' in var_0.doc)
    assert var_5 is True
    var_6 = 'func\\_with\\_underscores()'
    var_7 = bool('func\\_with\\_underscores()' in var_0.doc['test_module.func_with_underscores'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def documented_func():\n    """This is a docstring."""\n    pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.documented_func'
    var_5 = bool('test_module.documented_func' in var_0.docstring)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class MyClass:\n    @classmethod\n    def cls_method(cls): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.cls_method'
    var_7 = bool('test_module.MyClass.cls_method' in var_0.doc)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 8/42 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'CONST_VAR'
    var_4 = None
    var_5 = []
    var_6 = 42
    var_7 = []
    var_8 = 'int'
    var_9 = 'test_module.CONST_VAR'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/13 statements.
# Partially parsed test_class_api_with_members. Retrieved 7/19 statements.
# Partially parsed test_class_api_with_enum. Retrieved 8/25 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 7/19 statements.
# Partially parsed test_class_api_with_delete. Retrieved 7/23 statements.
# Partially parsed test_class_api_empty_class. Retrieved 7/12 statements.
# Partially parsed test_class_api_multiple_bases. Retrieved 6/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.MyClass'
    var_6 = 'test_module.MyClass'
    var_7 = bool('test_module.MyClass' in var_0.doc)
    assert var_7 is True
    var_8 = 'BaseClass'
    var_9 = bool('BaseClass' in var_0.doc['test_module.MyClass'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member1'
    var_3 = 'int'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'test_module'
    var_8 = 'test_module.MyClass'
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'member1'
    var_12 = bool('member1' in var_0.doc['test_module.MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'enum'
    var_2 = []
    var_3 = 'Enum'
    var_4 = []
    var_5 = 'MEMBER'
    var_6 = 'int'
    var_7 = []
    var_8 = 1
    var_9 = []
    var_10 = 'test_module'
    var_11 = 'test_module.MyEnum'
    var_12 = 'test_module.MyEnum'
    var_13 = bool('test_module.MyEnum' in var_0.doc)
    assert var_13 is True
    var_14 = 'Enums'
    var_15 = bool('Enums' in var_0.doc['test_module.MyEnum'])
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'int'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'test_module'
    var_8 = 'test_module.MyClass'
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = '_private'
    var_12 = bool('_private' not in var_0.doc['test_module.MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'member'
    var_3 = 'int'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'test_module'
    var_8 = 'test_module.MyClass'
    var_9 = 'test_module.MyClass'
    var_10 = bool('test_module.MyClass' in var_0.doc)
    assert var_10 is True
    var_11 = 'member'
    var_12 = bool('member' not in var_0.doc['test_module.MyClass'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'test_module'
    var_4 = 'test_module.EmptyClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True
    var_8 = var_0.doc[var_4]

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Base1'
    var_2 = []
    var_3 = 'Base2'
    var_4 = []
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_module.MyClass'
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.doc)
    assert var_9 is True
    var_10 = 'Base1'
    var_11 = bool('Base1' in var_0.doc['test_module.MyClass'])
    assert var_11 is True
    var_12 = 'Base2'
    var_13 = bool('Base2' in var_0.doc['test_module.MyClass'])
    assert var_13 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_const_type_with_constant_int. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_str. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_float. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_constant_bool. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_of_ints. Retrieved 3/10 statements.
# Partially parsed test_const_type_with_list_of_strings. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_tuple_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_set_of_ints. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_dict_of_str_to_int. Retrieved 4/13 statements.
# Partially parsed test_const_type_with_list_mixed_types. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_const_type_with_list_containing_non_constant. Retrieved 2/8 statements.
# Partially parsed test_const_type_with_call_to_int. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_str. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_call_to_list. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_unknown_call. Retrieved 3/7 statements.
# Partially parsed test_const_type_with_name_node. Retrieved 1/4 statements.


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
    var_0 = True
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
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'b'
    var_3 = []
    var_4 = 1
    var_5 = []
    var_6 = 2
    var_7 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = []

def test_case_0():
    var_0 = 'int'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'str'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'list'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'unknown_func'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_compile_magic_method_continues. Retrieved 5/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that line 15 predicate (is_magic(name)) evaluates to True and continues.'
    var_1 = False
    var_2 = 1
    var_3 = module_0.Parser(var_1, var_2, var_1)
    var_4 = var_3.compile()
    var_5 = '__init__'
    var_6 = bool('__init__' not in var_4)
    assert var_6 is True
    var_7 = '__str__'
    var_8 = bool('__str__' not in var_4)
    assert var_8 is True
    var_9 = 'regular_func'
    var_10 = bool('regular_func' in var_4)
    assert var_10 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_attr_single_level_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_none_in_chain. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_with_multiple_nested_levels. Retrieved 2/11 statements.
# Partially parsed test_attr_attribute_is_zero. Retrieved 2/5 statements.
# Partially parsed test_attr_attribute_is_false. Retrieved 2/5 statements.
# Partially parsed test_attr_attribute_is_empty_string. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'x'

def test_case_0():
    var_0 = 100
    var_1 = 'b.c.z'

def test_case_0():
    var_0 = 10
    var_1 = 'y'

def test_case_0():
    var_0 = 20
    var_1 = 'b.x.z'

def test_case_0():
    var_0 = None
    var_1 = 'b.c.d'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'deep'
    var_1 = 'b.c.d.value'

def test_case_0():
    var_0 = 0
    var_1 = 'x'

def test_case_0():
    var_0 = False
    var_1 = 'x'

def test_case_0():
    var_0 = ''
    var_1 = 'x'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_bases. Retrieved 10/14 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/9 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/14 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 11/15 statements.
# Partially parsed test_class_api_mixed_members_and_enums. Retrieved 10/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True
    var_12 = 'Members'
    var_13 = bool('Members' in var_0.doc['test_module.TestClass'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport enum\nclass Color(enum.Enum):\n    RED = 1\n    GREEN = 2\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 1
    var_4 = var_2.body[var_3]
    var_5 = var_4.bases
    var_6 = 'test_module'
    var_7 = 'test_module.Color'
    var_8 = var_4.body
    var_9 = var_0.class_api(var_6, var_7, var_5, var_8)
    var_10 = 'test_module.Color'
    var_11 = bool('test_module.Color' in var_0.doc)
    assert var_11 is True
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc['test_module.Color'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Parent:\n    pass\n\nclass Child(Parent):\n    pass\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 1
    var_4 = var_2.body[var_3]
    var_5 = var_4.bases
    var_6 = 'test_module'
    var_7 = 'test_module.Child'
    var_8 = var_4.body
    var_9 = var_0.class_api(var_6, var_7, var_5, var_8)
    var_10 = 'test_module.Child'
    var_11 = bool('test_module.Child' in var_0.doc)
    assert var_11 is True
    var_12 = 'Bases'
    var_13 = bool('Bases' in var_0.doc['test_module.Child'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'test_module.EmptyClass'
    var_7 = bool('test_module.EmptyClass' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    attr: int\n    del attr\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.TestClass'
    var_7 = []
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.TestClass'
    var_11 = bool('test_module.TestClass' in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass TestClass:\n    value = 42  # type: int\n    '
    var_2 = True
    var_3 = module_1.parse(var_1, type_comments=var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test_module'
    var_7 = 'test_module.TestClass'
    var_8 = []
    var_9 = var_5.body
    var_10 = var_0.class_api(var_6, var_7, var_8, var_9)
    var_11 = 'test_module.TestClass'
    var_12 = bool('test_module.TestClass' in var_0.doc)
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport enum\nclass Status(enum.Enum):\n    ACTIVE = 1\n    INACTIVE = 2\n    '
    var_2 = module_1.parse(var_1)
    var_3 = 1
    var_4 = var_2.body[var_3]
    var_5 = 'test_module'
    var_6 = 'test_module.Status'
    var_7 = var_4.bases
    var_8 = var_4.body
    var_9 = var_0.class_api(var_5, var_6, var_7, var_8)
    var_10 = 'test_module.Status'
    var_11 = bool('test_module.Status' in var_0.doc)
    assert var_11 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 6/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = None
    var_8 = 0



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_visit_name_self_ty_replacement. Retrieved 4/8 statements.
# Partially parsed test_visit_name_no_alias. Retrieved 4/8 statements.
# Partially parsed test_visit_name_with_alias_simple. Retrieved 6/10 statements.
# Partially parsed test_visit_name_with_alias_circular_reference. Retrieved 5/9 statements.
# Partially parsed test_visit_name_typevar_alias. Retrieved 8/12 statements.
# Partially parsed test_visit_name_with_nested_root. Retrieved 6/10 statements.
# Partially parsed test_visit_name_empty_root. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'MyType'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'SomeName'
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyName'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyName'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.MyName'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    var_4 = 'MyName'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = 'module.TypeVar'
    var_3 = "TypeVar('T')"
    var_4 = 'typing.TypeVar'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'T'
    var_8 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'package.module'
    var_1 = 'package.module.Item'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Item'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'Item'
    var_2 = 'str'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = []



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_func_api_simple_function. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_self. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 6/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 7/17 statements.
# Partially parsed test_func_api_no_annotations. Retrieved 7/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(x: int, y: str) -> bool: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '|'
    var_10 = bool('|' in var_0.doc['test_module.func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def func(x: int = 5, y: str = 'test') -> bool: pass"
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '|'
    var_10 = bool('|' in var_0.doc['test_module.func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method(self, x: int) -> str: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Class.method'
    var_5 = True
    var_6 = False
    var_7 = 'test_module.Class.method'
    var_8 = bool('test_module.Class.method' in var_0.doc)
    assert var_8 is True
    var_9 = 'Self'
    var_10 = bool('Self' in var_0.doc['test_module.Class.method'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method(cls, x: int) -> str: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.Class.method'
    var_5 = True
    var_6 = 'test_module.Class.method'
    var_7 = bool('test_module.Class.method' in var_0.doc)
    assert var_7 is True
    var_8 = 'type[Self]'
    var_9 = bool('type[Self]' in var_0.doc['test_module.Class.method'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(*args, **kwargs) -> None: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '|'
    var_10 = bool('|' in var_0.doc['test_module.func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(x: int, *, y: str) -> bool: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '|'
    var_10 = bool('|' in var_0.doc['test_module.func'])
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(x, y): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = 'Any'
    var_10 = bool('Any' in var_0.doc['test_module.func'])
    assert var_10 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_visit_name_predicate_line_6_true. Retrieved 14/35 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True.'
    var_1 = 'module'
    var_2 = 'module.MyType'
    var_3 = 'int'
    var_4 = {var_2: var_3}
    var_5 = ''
    var_6 = module_0.Resolver(var_1, var_4, var_5)
    var_7 = 'MyType'
    var_8 = []
    var_9 = 'builtins'
    var_10 = __import__(var_9)
    var_11 = '_m'
    var_12 = 'test_module'
    var_13 = None
    var_14 = '_m'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_class_api_predicate_line_36. Retrieved 14/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'enum_member'
    var_4 = None
    var_5 = []
    var_6 = 0
    var_7 = 'enum.Enum'
    var_8 = module_1.parse(var_7)
    var_9 = var_8.body[var_6]
    var_10 = var_9.value
    var_11 = [var_10]
    var_12 = [var_3]
    var_13 = 'enum_member'
    var_14 = var_13 in var_12
    assert var_14 is True



# Parsed testcases at query #37
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'x = 5'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body
    var_7 = var_0.class_api(var_1, var_2, var_3, var_6)
    var_8 = bool(var_2 in var_0.doc)
    assert var_8 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_func_ann_with_self_annotation. Retrieved 15/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'self'
    var_3 = 0
    var_4 = 'int'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = []
    var_9 = 'x'
    var_10 = 'str'
    var_11 = module_1.parse(var_10)
    var_12 = var_11.body[var_3]
    var_13 = var_12.value
    var_14 = []
    var_15 = True
    var_16 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_predicate_line_32_evaluates_to_false. Retrieved 10/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = 'del obj.attr'
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = [var_7]
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = var_0.doc[var_2]
    assert var_10 == ''



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_independent_dict_instances. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = var_2.link
    assert var_3 is True
    var_4 = var_2.toc
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = module_0.Parser()
    var_2 = 'test'
    var_3 = bool('test' not in var_1.doc)
    assert var_3 is True
    var_4 = var_0.doc
    var_5 = bool(var_0.doc is not var_1.doc)
    assert var_5 is True
    var_6 = var_0.level
    var_7 = bool(var_0.level is not var_1.level)
    assert var_7 is True
    var_8 = var_0.imp
    var_9 = bool(var_0.imp is not var_1.imp)
    assert var_9 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_api_function_def. Retrieved 8/12 statements.
# Partially parsed test_api_async_function_def. Retrieved 8/12 statements.
# Partially parsed test_api_class_def. Retrieved 8/12 statements.
# Partially parsed test_api_with_decorators. Retrieved 8/12 statements.
# Partially parsed test_api_with_prefix. Retrieved 9/13 statements.
# Partially parsed test_api_with_link_false. Retrieved 8/12 statements.
# Partially parsed test_api_with_docstring. Retrieved 8/12 statements.
# Partially parsed test_api_nested_class. Retrieved 8/12 statements.
# Partially parsed test_api_underscore_escaping. Retrieved 8/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\ndef example_func():\n    '''Example function'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.example_func'
    var_9 = bool('test_module.example_func' in var_2.doc)
    assert var_9 is True
    var_10 = 'example_func()'
    var_11 = bool('example_func()' in var_2.doc['test_module.example_func'])
    assert var_11 is True
    var_12 = 'Full name:'
    var_13 = bool('Full name:' in var_2.doc['test_module.example_func'])
    assert var_13 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\nasync def async_func():\n    '''Async function'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.async_func'
    var_9 = bool('test_module.async_func' in var_2.doc)
    assert var_9 is True
    var_10 = 'async async_func()'
    var_11 = bool('async async_func()' in var_2.doc['test_module.async_func'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\nclass ExampleClass:\n    '''Example class'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.ExampleClass'
    var_9 = bool('test_module.ExampleClass' in var_2.doc)
    assert var_9 is True
    var_10 = 'class ExampleClass'
    var_11 = bool('class ExampleClass' in var_2.doc['test_module.ExampleClass'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\n@staticmethod\ndef decorated_func():\n    '''Decorated function'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.decorated_func'
    var_9 = bool('test_module.decorated_func' in var_2.doc)
    assert var_9 is True
    var_10 = 'Decorators'
    var_11 = bool('Decorators' in var_2.doc['test_module.decorated_func'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\ndef method():\n    '''Method'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = 'OuterClass'
    var_8 = var_2.api(var_3, var_6, prefix=var_7)
    var_9 = 'test_module.OuterClass.method'
    var_10 = bool('test_module.OuterClass.method' in var_2.doc)
    assert var_10 is True
    var_11 = var_2.level['test_module.OuterClass.method']
    assert var_11 == 0

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = 'test_module'
    var_4 = "\ndef func():\n    '''Function'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_0]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_2.doc)
    assert var_9 is True
    var_10 = '<a id='
    var_11 = bool('<a id=' not in var_2.doc['test_module.func'])
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = '\ndef example():\n    """Example function.\n    \n    >>> example()\n    """\n    pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.example'
    var_9 = bool('test_module.example' in var_2.docstring)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = '\nclass OuterClass:\n    class InnerClass:\n        pass\n'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.OuterClass'
    var_9 = bool('test_module.OuterClass' in var_2.doc)
    assert var_9 is True
    var_10 = 'test_module.OuterClass.InnerClass'
    var_11 = bool('test_module.OuterClass.InnerClass' in var_2.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    var_3 = 'test_module'
    var_4 = "\ndef func_with_underscores():\n    '''Function with underscores'''\n    pass\n"
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_1]
    var_7 = var_2.api(var_3, var_6)
    var_8 = 'test_module.func_with_underscores'
    var_9 = bool('test_module.func_with_underscores' in var_2.doc)
    assert var_9 is True
    var_10 = 'func\\_with\\_underscores'
    var_11 = bool('func\\_with\\_underscores' in var_2.doc['test_module.func_with_underscores'])
    assert var_11 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 2/3 statements.
# Partially parsed test_parser_new_classmethod_with_false_values. Retrieved 2/3 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #44
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a regular line\nAnother regular line'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a regular line\nAnother regular line'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n>>> y = 2\n>>> z = x + y'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n>>> y = 2\n>>> z = x + y\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Some text\n>>> x = 1\n>>> y = 2\nMore text'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Some text\n```python\n>>> x = 1\n>>> y = 2\n```\nMore text'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\ntext\n>>> y = 2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\ntext\n```python\n>>> y = 2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\nSome text'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\nSome text'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Some text\n>>> x = 1'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Some text\n```python\n>>> x = 1\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> a = 1\n>>> b = 2\n\n>>> c = 3'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> a = 1\n>>> b = 2\n```\n\n```python\n>>> c = 3\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = '>>> x = 1\n1\n>>> y = 2'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '```python\n>>> x = 1\n```\n1\n```python\n>>> y = 2\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Just a comment'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Just a comment'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 12/16 statements.
# Partially parsed test_class_api_with_enums. Retrieved 14/16 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 11/14 statements.
# Partially parsed test_class_api_with_bases. Retrieved 11/14 statements.
# Partially parsed test_class_api_empty_body. Retrieved 6/7 statements.
# Partially parsed test_class_api_with_type_comment. Retrieved 10/12 statements.
# Partially parsed test_class_api_mixed_public_private. Retrieved 12/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.body
    var_9 = 'int'
    var_10 = 'str'
    var_11 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_12 = bool(var_2 in var_0.doc)
    assert var_12 is True
    var_13 = 'Members'
    var_14 = bool('Members' in var_0.doc[var_2])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'enum.Enum'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = [var_7]
    var_9 = '\nclass TestEnum(enum.Enum):\n    MEMBER1: int\n    MEMBER2: str\n    '
    var_10 = module_1.parse(var_9)
    var_11 = var_10.body[var_3]
    var_12 = var_11.body
    var_13 = var_0.class_api(var_1, var_2, var_8, var_12)
    var_14 = bool(var_2 in var_0.doc)
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '\nclass TestClass:\n    public_attr: int\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.body
    var_9 = 'int'
    var_10 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_11 = bool(var_2 in var_0.doc)
    assert var_11 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = '\nclass BaseClass:\n    pass\n\nclass TestClass(BaseClass):\n    pass\n    '
    var_4 = module_1.parse(var_3)
    var_5 = 1
    var_6 = var_4.body[var_5]
    var_7 = var_6.bases
    var_8 = var_6.body
    var_9 = 'BaseClass'
    var_10 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_11 = bool(var_2 in var_0.doc)
    assert var_11 is True
    var_12 = 'Bases'
    var_13 = bool('Bases' in var_0.doc[var_2])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = bool(var_2 in var_0.doc)
    assert var_6 is True
    var_7 = 'Bases'
    var_8 = bool('Bases' not in var_0.doc[var_2])
    assert var_8 is True
    var_9 = 'Members'
    var_10 = bool('Members' not in var_0.doc[var_2])
    assert var_10 is True
    var_11 = 'Enums'
    var_12 = bool('Enums' not in var_0.doc[var_2])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '\nclass TestClass:\n    attr = 42  # type: int\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.body
    var_9 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_10 = bool(var_2 in var_0.doc)
    assert var_10 is True
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc[var_2])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '\nclass TestClass:\n    public_field: int\n    _private_field: str\n    '
    var_5 = module_1.parse(var_4)
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_7.body
    var_9 = 'int'
    var_10 = 'str'
    var_11 = var_0.class_api(var_1, var_2, var_3, var_8)
    var_12 = bool(var_2 in var_0.doc)
    assert var_12 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_class_api_predicate_line_19_false. Retrieved 13/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False when len(node.targets) != 1'
    var_1 = module_0.Parser()
    var_2 = 'x = y = 5'
    var_3 = module_1.parse(var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_5.targets
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_5.targets
    var_9 = len(var_8)
    var_10 = 1
    var_11 = var_9 == var_10
    var_12 = var_5.targets[var_4]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_attr_predicate_at_line_4_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'nested.value'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 15/33 statements.
# Partially parsed test_class_api_with_enum. Retrieved 14/39 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 8/23 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 11/28 statements.
# Partially parsed test_class_api_with_bases. Retrieved 5/12 statements.
# Partially parsed test_class_api_empty_class. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = []
    var_3 = []
    var_4 = 'attr1'
    var_5 = 'str'
    var_6 = []
    var_7 = 'default'
    var_8 = []
    var_9 = 1
    var_10 = 'attr2'
    var_11 = 42
    var_12 = []
    var_13 = None
    var_14 = []
    var_15 = 'test_module'
    var_16 = 'test_module.TestClass'
    var_17 = []
    var_18 = bool('test_module.TestClass' in var_0.doc or True)
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Color'
    var_2 = 'enum'
    var_3 = []
    var_4 = 'Enum'
    var_5 = []
    var_6 = []
    var_7 = 'RED'
    var_8 = 'str'
    var_9 = []
    var_10 = 'red'
    var_11 = []
    var_12 = 1
    var_13 = 'BLUE'
    var_14 = []
    var_15 = 'blue'
    var_16 = []
    var_17 = []
    var_18 = 'test_module'
    var_19 = 'test_module.Color'
    var_20 = bool('test_module.Color' in var_0.doc or True)
    assert var_20 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'attr1'
    var_2 = 'int'
    var_3 = []
    var_4 = 10
    var_5 = []
    var_6 = 1
    var_7 = 'test_module'
    var_8 = 'test_module.TestClass'
    var_9 = []
    var_10 = bool('test_module.TestClass' in var_0.doc or True)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'str'
    var_3 = []
    var_4 = 'secret'
    var_5 = []
    var_6 = 1
    var_7 = 'public'
    var_8 = 'int'
    var_9 = []
    var_10 = 5
    var_11 = []
    var_12 = 'test_module'
    var_13 = 'test_module.TestClass'
    var_14 = []
    var_15 = bool('test_module.TestClass' in var_0.doc or True)
    assert var_15 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = []
    var_4 = 'test_module'
    var_5 = 'test_module.DerivedClass'
    var_6 = bool('test_module.DerivedClass' in var_0.doc or True)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = bool('test_module.EmptyClass' in var_0.doc or True)
    assert var_6 is True



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_e_type_empty_elements.
# Partially parsed test_e_type_single_element_with_constants. Retrieved 3/10 statements.
# Partially parsed test_e_type_multiple_elements_same_type. Retrieved 4/13 statements.
# Partially parsed test_e_type_multiple_elements_different_types. Retrieved 4/13 statements.
# Partially parsed test_e_type_mixed_types_in_same_element. Retrieved 2/8 statements.
# Partially parsed test_e_type_none_element. Retrieved 2/7 statements.
# Partially parsed test_e_type_empty_sequence_element. Retrieved 2/4 statements.
# Partially parsed test_e_type_non_constant_in_element. Retrieved 3/9 statements.
# Partially parsed test_e_type_float_constants. Retrieved 2/8 statements.
# Partially parsed test_e_type_string_constants. Retrieved 2/8 statements.
# Partially parsed test_e_type_bool_constants. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 2
    var_3 = []
    var_4 = 3
    var_5 = []
    var_6 = 4
    var_7 = []

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
    var_0 = 1
    var_1 = []
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []

def test_case_0():
    var_0 = 1.5
    var_1 = []
    var_2 = 2.5
    var_3 = []

def test_case_0():
    var_0 = 'hello'
    var_1 = []
    var_2 = 'world'
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = False
    var_3 = []



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 10/17 statements.


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
    var_8 = 'test_module'
    var_9 = 'test_module.test_func'
    var_10 = False
    var_11 = 'test_module.test_func'
    var_12 = bool('test_module.test_func' in var_0.doc)
    assert var_12 is True
    var_13 = '/'
    var_14 = bool('/' in var_0.doc['test_module.test_func'])
    assert var_14 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_globals_predicate_line_38_false. Retrieved 9/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to False.'
    var_1 = module_0.Parser()
    var_2 = '__all__'
    var_3 = None
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 2
    var_8 = []
    var_9 = 'test_module'
    var_10 = var_1.imp[var_9]
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_visit_name_self_ty_match. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_class_api_line_25_predicate_false. Retrieved 7/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private_attr'
    var_2 = None
    var_3 = []
    var_4 = 42
    var_5 = []
    var_6 = 'test_module'
    var_7 = 'test_class'
    var_8 = []
    var_9 = var_0.doc['test_class']
    assert var_9 == ''



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_class_api_is_enum_predicate. Retrieved 14/35 statements.


import ast as module_0

def test_case_0():
    var_0 = '\nclass MyEnum:\n    VALUE = 1\n'
    var_1 = module_0.parse(var_0)
    var_2 = 0
    var_3 = var_1.body[var_2]
    var_4 = 'enum.Enum'
    var_5 = [var_4]
    var_6 = 'enum.'
    var_7 = lambda s: s.startswith(var_6)
    var_8 = map(var_7, var_5)
    var_9 = any(var_8)
    assert var_9 is True
    var_10 = var_3.body[var_2]
    var_11 = var_10.targets
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_10.targets[var_2]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_globals_ann_assign_with_value. Retrieved 4/9 statements.
# Partially parsed test_globals_assign_uppercase_constant. Retrieved 4/9 statements.
# Partially parsed test_globals_assign_with_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_all_filter. Retrieved 4/10 statements.
# Partially parsed test_globals_ignores_non_simple_assign. Retrieved 4/9 statements.
# Partially parsed test_globals_ann_assign_without_value. Retrieved 4/9 statements.
# Partially parsed test_globals_lowercase_variable. Retrieved 4/9 statements.
# Partially parsed test_globals_multiple_targets_ignored. Retrieved 4/9 statements.
# Partially parsed test_globals_string_constant_type. Retrieved 4/9 statements.
# Partially parsed test_globals_list_constant_type. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int = 42'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.x']
    assert var_6 == '42'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'MAX_VALUE = 100'
    var_3 = 0
    var_4 = 'test_module.MAX_VALUE'
    var_5 = bool('test_module.MAX_VALUE' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.MAX_VALUE'
    var_7 = bool('test_module.MAX_VALUE' in var_0.const)
    assert var_7 is True
    var_8 = var_0.const['test_module.MAX_VALUE']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = 5  # type: int'
    var_3 = True
    var_4 = 0
    var_5 = 'test_module.x'
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.alias['test_module.x']
    assert var_7 == '5'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "__all__ = ['func1', 'func2']"
    var_3 = 0
    var_4 = 'test_module.func1'
    var_5 = bool('test_module.func1' in var_0.imp[var_1])
    assert var_5 is True
    var_6 = 'test_module.func2'
    var_7 = bool('test_module.func2' in var_0.imp[var_1])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x, y = 1, 2'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x: int'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'my_var = 42'
    var_3 = 0
    var_4 = 'test_module.my_var'
    var_5 = bool('test_module.my_var' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.my_var'
    var_7 = bool('test_module.my_var' not in var_0.const)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'x = y = 5'
    var_3 = 0
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' not in var_0.alias)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = "MESSAGE = 'hello'"
    var_3 = 0
    var_4 = var_0.const['test_module.MESSAGE']
    assert var_4 == 'str'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'NUMBERS = [1, 2, 3]'
    var_3 = 0
    var_4 = 'test_module.NUMBERS'
    var_5 = bool('test_module.NUMBERS' in var_0.const)
    assert var_5 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/4 statements.
# Partially parsed test_parser_new_with_toc_true. Retrieved 3/4 statements.


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
    var_0 = 'test'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'doc'
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = False
    var_7 = module_0.Parser(var_5, var_5, var_6, var_2, var_4)
    var_8 = var_7.level
    var_9 = bool(var_7.level == var_2)
    assert var_9 is True
    var_10 = var_7.doc
    var_11 = bool(var_7.doc == var_4)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_with_toc_sets_link. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = True
    var_2 = module_0.Parser(var_0, toc=var_1)
    var_3 = var_2.link
    assert var_3 is True
    var_4 = var_2.toc
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Parser(var_0, toc=var_0)
    var_2 = var_1.link
    assert var_2 is False
    var_3 = var_1.toc
    assert var_3 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_classmethod_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_load_docstring_with_valid_doc. Retrieved 7/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.func'
    var_3 = 'Module test_module'
    var_4 = 'Function func'
    var_5 = 'This is a test docstring'
    var_6 = 'test_module'
    var_7 = 'test_module'
    var_8 = bool('test_module' in var_0.docstring)
    assert var_8 is True
    var_9 = var_0.docstring['test_module']
    assert var_9 == 'processed docstring'
    var_10 = 'test_module.func'
    var_11 = bool('test_module.func' in var_0.docstring)
    assert var_11 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_api_function_def. Retrieved 4/15 statements.
# Partially parsed test_api_async_function_def. Retrieved 4/15 statements.
# Partially parsed test_api_class_def. Retrieved 4/15 statements.
# Partially parsed test_api_with_decorators. Retrieved 4/15 statements.
# Partially parsed test_api_with_prefix. Retrieved 5/16 statements.
# Partially parsed test_api_with_link. Retrieved 4/15 statements.
# Partially parsed test_api_nested_class_methods. Retrieved 4/15 statements.
# Partially parsed test_api_with_underscore_escaping. Retrieved 4/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def example_func(): pass'
    var_4 = 'test_module.example_func'
    var_5 = bool('test_module.example_func' in var_0.doc)
    assert var_5 is True
    var_6 = '## example_func()'
    var_7 = bool('## example_func()' in var_0.doc['test_module.example_func'])
    assert var_7 is True
    var_8 = '*Full name:* `test_module.example_func`'
    var_9 = bool('*Full name:* `test_module.example_func`' in var_0.doc['test_module.example_func'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'async def async_func(): pass'
    var_4 = 'test_module.async_func'
    var_5 = bool('test_module.async_func' in var_0.doc)
    assert var_5 is True
    var_6 = '## async async_func()'
    var_7 = bool('## async async_func()' in var_0.doc['test_module.async_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'class ExampleClass: pass'
    var_4 = 'test_module.ExampleClass'
    var_5 = bool('test_module.ExampleClass' in var_0.doc)
    assert var_5 is True
    var_6 = '## class ExampleClass'
    var_7 = bool('## class ExampleClass' in var_0.doc['test_module.ExampleClass'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = '@staticmethod\ndef decorated_func(): pass'
    var_4 = 'test_module.decorated_func'
    var_5 = bool('test_module.decorated_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'Decorators'
    var_7 = bool('Decorators' in var_0.doc['test_module.decorated_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def method_name(): pass'
    var_4 = 'ClassName'
    var_5 = 'test_module.ClassName.method_name'
    var_6 = bool('test_module.ClassName.method_name' in var_0.doc)
    assert var_6 is True
    var_7 = '### method_name()'
    var_8 = bool('### method_name()' in var_0.doc['test_module.ClassName.method_name'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def func_name(): pass'
    var_4 = 'test_module.func_name'
    var_5 = bool('test_module.func_name' in var_0.doc)
    assert var_5 is True
    var_6 = '<a id='
    var_7 = bool('<a id=' in var_0.doc['test_module.func_name'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = '\nclass MyClass:\n    def method1(self): pass\n    def method2(self): pass\n'
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method1'
    var_7 = bool('test_module.MyClass.method1' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.MyClass.method2'
    var_9 = bool('test_module.MyClass.method2' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 0
    var_3 = 'def func_with_underscores(): pass'
    var_4 = 'test_module.func_with_underscores'
    var_5 = bool('test_module.func_with_underscores' in var_0.doc)
    assert var_5 is True
    var_6 = 'func\\_with\\_underscores()'
    var_7 = bool('func\\_with\\_underscores()' in var_0.doc['test_module.func_with_underscores'])
    assert var_7 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_api_predicate_line_21_evaluates_to_true. Retrieved 11/38 statements.


def test_case_0():
    var_0 = 'test_func'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'test_decorator'
    var_9 = []
    var_10 = None
    var_11 = '@'
    var_12 = 'test_root'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_parser_new_classmethod. Retrieved 3/5 statements.
# Partially parsed test_parser_new_with_toc_true. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = module_0.Parser(var_0, var_1, var_2)
    var_4 = var_3.link
    assert var_4 is True
    var_5 = var_3.toc
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    var_3 = var_2.link
    assert var_3 is False
    var_4 = var_2.toc
    assert var_4 is False

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_func_api_with_defaults. Retrieved 7/17 statements.
# Partially parsed test_func_api_with_self. Retrieved 7/17 statements.
# Partially parsed test_func_api_classmethod. Retrieved 6/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 7/17 statements.
# Partially parsed test_func_api_no_annotations. Retrieved 8/17 statements.
# Partially parsed test_func_api_with_kwonly_args. Retrieved 7/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def func(a: int, b: str = 'default') -> bool: pass"
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '| a |'
    var_10 = bool('| a |' in var_0.doc['test_module.func'])
    assert var_10 is True
    var_11 = '| b |'
    var_12 = bool('| b |' in var_0.doc['test_module.func'])
    assert var_12 is True
    var_13 = '| return |'
    var_14 = bool('| return |' in var_0.doc['test_module.func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method(self, a: int) -> str: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.method'
    var_5 = True
    var_6 = False
    var_7 = 'test_module.method'
    var_8 = bool('test_module.method' in var_0.doc)
    assert var_8 is True
    var_9 = '| self |'
    var_10 = bool('| self |' in var_0.doc['test_module.method'])
    assert var_10 is True
    var_11 = '| a |'
    var_12 = bool('| a |' in var_0.doc['test_module.method'])
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def method(cls, x: int) -> None: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.method'
    var_5 = True
    var_6 = 'test_module.method'
    var_7 = bool('test_module.method' in var_0.doc)
    assert var_7 is True
    var_8 = 'type[Self]'
    var_9 = bool('type[Self]' in var_0.doc['test_module.method'])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(a: int, *args: str, **kwargs: bool) -> None: pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '| a |'
    var_10 = bool('| a |' in var_0.doc['test_module.func'])
    assert var_10 is True
    var_11 = '*args'
    var_12 = bool('*args' in var_0.doc['test_module.func'])
    assert var_12 is True
    var_13 = '**kwargs'
    var_14 = bool('**kwargs' in var_0.doc['test_module.func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def func(a, b): pass'
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = None
    var_6 = False
    var_7 = False
    var_8 = 'test_module.func'
    var_9 = bool('test_module.func' in var_0.doc)
    assert var_9 is True
    var_10 = '| a |'
    var_11 = bool('| a |' in var_0.doc['test_module.func'])
    assert var_11 is True
    var_12 = '| b |'
    var_13 = bool('| b |' in var_0.doc['test_module.func'])
    assert var_13 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "def func(a: int, *, b: str = 'x') -> None: pass"
    var_2 = 0
    var_3 = 'test_module'
    var_4 = 'test_module.func'
    var_5 = False
    var_6 = False
    var_7 = 'test_module.func'
    var_8 = bool('test_module.func' in var_0.doc)
    assert var_8 is True
    var_9 = '| a |'
    var_10 = bool('| a |' in var_0.doc['test_module.func'])
    assert var_10 is True
    var_11 = '| b |'
    var_12 = bool('| b |' in var_0.doc['test_module.func'])
    assert var_12 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 7/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 23 evaluates to False when type_comment is not None.'
    var_1 = module_0.Parser()
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'int'
    var_8 = 'test_module'



