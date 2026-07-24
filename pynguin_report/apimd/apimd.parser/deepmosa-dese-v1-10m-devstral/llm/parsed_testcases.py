####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_imports_with_import_node. Retrieved 4/8 statements.
# Partially parsed test_imports_with_import_node_and_asname. Retrieved 4/8 statements.
# Partially parsed test_imports_with_import_from_node_and_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_import_from_node_and_asname. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = 'dd'
    var_5 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_docstring. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg.sub.func'
    var_3 = '...'
    var_4 = None



# Parsed testcases at query #3
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = 1
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_1._defaults(var_4)
    var_6 = list(var_5)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a&b'
    var_1 = module_0.Constant()
    var_2 = [var_1]
    var_3 = module_1._defaults(var_2)
    var_4 = list(var_3)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a|b'
    var_1 = module_0.Constant()
    var_2 = [var_1]
    var_3 = module_1._defaults(var_2)
    var_4 = list(var_3)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = None
    var_1 = 'x&y'
    var_2 = module_0.Constant()
    var_3 = 'z'
    var_4 = module_0.Name()
    var_5 = [var_0, var_2, var_4]
    var_6 = module_1._defaults(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_visit_Constant_with_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_invalid_string. Retrieved 4/6 statements.
# Partially parsed test_visit_Constant_with_valid_name. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_public_with_magic_name. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_private_name_not_in_all. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_public_name_in_all. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_child_in_all. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 7/11 statements.
# Partially parsed test_is_public_with_const_not_in_all. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = '__init__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = '_private'
    var_4 = var_0.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent.child'
    var_3 = {var_2}
    var_4 = ''
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'parent.child'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'CONST'
    var_4 = 'int'
    var_5 = var_0.is_public(var_3)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_docstring_skips_non_matching_names. Retrieved 11/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.submodule'
    var_2 = 'other.module'
    var_3 = ''
    var_4 = 'module'
    var_5 = 'm'
    var_6 = ()
    var_7 = 'submodule'
    var_8 = {var_7: var_3}
    var_9 = type(var_5, var_6, var_8)
    var_10 = var_0.load_docstring(var_4, var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 16/29 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 'attr1'
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = None
    var_12 = 'attr2'
    var_13 = 42
    var_14 = module_1.Constant()
    var_15 = 'attr3'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 'OPTION1'
    var_8 = None
    var_9 = 'OPTION2'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_public_with_child_in_doc. Retrieved 8/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = {var_2}
    var_4 = 'pkg.subpkg'
    var_5 = 'pkg.subpkg.module'
    var_6 = ''
    var_7 = var_0.is_public(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_multiple_levels. Retrieved 1/8 statements.
# Partially parsed test_attr_with_none_intermediate. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'simple'

def test_case_0():
    var_0 = 'nested.attr'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'nested.nonexistent'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'level1.level2.level3'

def test_case_0():
    var_0 = 'level1.level2'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_walk_body_single_statement.
# Failed to parse test_walk_body_multiple_statements.
# Partially parsed test_walk_body_if_statement. Retrieved 2/13 statements.
# Partially parsed test_walk_body_nested_if_statements. Retrieved 4/20 statements.
# Failed to parse test_walk_body_try_statement.
# Partially parsed test_walk_body_try_with_multiple_handlers. Retrieved 2/19 statements.
# Partially parsed test_walk_body_mixed_statements. Retrieved 5/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()

import ast as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = module_0.Name()
    var_2 = 'x'
    var_3 = module_0.Name()

def test_case_0():
    var_0 = []
    var_1 = []

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = []
    var_4 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_public_with_all_listed. Retrieved 15/18 statements.
# Partially parsed test_is_public_without_all_listed. Retrieved 8/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'root.sub1'
    var_3 = 'root.sub2'
    var_4 = {var_2, var_3}
    var_5 = 'root.sub1.item1'
    var_6 = 'root.sub1.item2'
    var_7 = {var_5, var_6}
    var_8 = ''
    var_9 = var_0.is_public(var_2)
    var_10 = var_0.is_public(var_5)
    var_11 = var_0.is_public(var_6)
    var_12 = var_0.is_public(var_3)
    var_13 = 'root.sub3'
    var_14 = var_0.is_public(var_13)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.public'
    var_4 = 'root._private'
    var_5 = ''
    var_6 = var_0.is_public(var_3)
    var_7 = var_0.is_public(var_4)



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__attr_returns_none_when_intermediate_attribute_is_none. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'a.b'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_class_api_with_enum_members. Retrieved 13/21 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 12/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 9/18 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = []
    var_8 = var_0.class_api(var_1, var_2, var_6, var_7)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'Enum'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 'MEMBER1'
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = 'MEMBER2'
    var_11 = 2
    var_12 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = 'another_attr'
    var_10 = 42
    var_11 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_class_api_with_members. Retrieved 10/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = module_1.Constant()



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = 'Base1'
    var_4 = module_1.Name()
    var_5 = 'Base2'
    var_6 = module_1.Name()
    var_7 = [var_4, var_6]
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = '| Bases |\n|:---:|\n| `Base1` | `Base2` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.EnumClass'
    var_3 = 'enum.Enum'
    var_4 = module_1.Name()
    var_5 = [var_4]
    var_6 = 'A'
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Name()
    var_10 = module_1.AnnAssign()
    var_11 = 'B'
    var_12 = module_1.Name()
    var_13 = module_1.Name()
    var_14 = module_1.AnnAssign()
    var_15 = [var_10, var_14]
    var_16 = var_0.class_api(var_1, var_2, var_5, var_15)
    var_17 = '| Enums |\n|:---:|\n| A |\n| B |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'attr1'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = module_1.AnnAssign()
    var_9 = 'attr2'
    var_10 = module_1.Name()
    var_11 = 'str'
    var_12 = module_1.Name()
    var_13 = module_1.AnnAssign()
    var_14 = [var_8, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_3, var_14)
    var_16 = '| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'attr1'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = module_1.AnnAssign()
    var_9 = module_1.Name()
    var_10 = [var_9]
    var_11 = module_1.Delete()
    var_12 = [var_8, var_11]
    var_13 = var_0.class_api(var_1, var_2, var_3, var_12)
    var_14 = ''

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'attr1'
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = module_1.Assign()
    var_10 = [var_9]
    var_11 = var_0.class_api(var_1, var_2, var_3, var_10)
    var_12 = '| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n\n'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__e_type_with_none_element. Retrieved 2/3 statements.
# Partially parsed test__e_type_with_non_constant_element. Retrieved 3/4 statements.
# Partially parsed test__e_type_with_single_constant_element. Retrieved 2/3 statements.
# Partially parsed test__e_type_with_multiple_same_type_constant_elements. Retrieved 4/5 statements.
# Partially parsed test__e_type_with_multiple_different_type_constant_elements. Retrieved 3/4 statements.
# Partially parsed test__e_type_with_multiple_sequences. Retrieved 6/7 statements.
# Partially parsed test__e_type_with_mixed_sequences. Retrieved 6/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]



# Parsed testcases at query #19
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = set()

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom sys import path\nx = 1\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nX = 1\nY: int = 2\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Foo:\n    """Class docstring."""\n    def bar(self):\n        pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n@decorator\ndef foo():\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_with_non_constant. Retrieved 7/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = module_1.Constant()
    var_4 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'foo'
    var_3 = module_1.Constant()
    var_4 = 'bar'
    var_5 = module_1.Constant()
    var_6 = [var_3, var_5]
    var_7 = module_1.List()
    var_8 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = 1
    var_3 = module_1.Constant()
    var_4 = 2
    var_5 = module_1.Constant()
    var_6 = 'module'



# Parsed testcases at query #21
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'from enum import Enum\nclass Test(Enum):\n    A = 1\n    B = 2'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #22
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = None
    var_2 = []
    var_3 = 0
    var_4 = module_1.ImportFrom()
    var_5 = 'root'
    var_6 = var_0.imports(var_5, var_4)
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #23
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'name'
    var_3 = []
    var_4 = module_1.Subscript()
    var_5 = [var_4]
    var_6 = module_1.Delete()
    var_7 = [var_6]
    var_8 = var_0.class_api(var_1, var_2, var_3, var_7)
    var_9 = var_0.doc[var_2]
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_docstring_updates_docstring_when_doc_is_not_none. Retrieved 16/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.submodule.func'
    var_2 = ''
    var_3 = 'MockModule'
    var_4 = ()
    var_5 = 'submodule'
    var_6 = 'MockSubmodule'
    var_7 = ()
    var_8 = 'func'
    var_9 = None
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = type(var_6, var_7, var_11)
    var_13 = {var_5: var_12}
    var_14 = type(var_3, var_4, var_13)
    var_15 = 'module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 10/17 statements.
# Partially parsed test_func_ann_with_cls_method_and_annotation. Retrieved 9/16 statements.
# Partially parsed test_func_ann_without_annotation. Retrieved 6/11 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 11/19 statements.
# Partially parsed test_func_ann_with_self_ty_resolution. Retrieved 10/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = 'root'
    var_8 = True
    var_9 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'type[Self]'
    var_3 = module_1.Load()
    var_4 = 'x'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = 'root'
    var_8 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = '*'
    var_5 = None
    var_6 = 'y'
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = 'root'
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = module_1.Load()
    var_4 = 'x'
    var_5 = module_1.Load()
    var_6 = 'root'
    var_7 = True
    var_8 = False
    var_9 = 'root.Parent'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_true. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_func_ann_yields_type_self_when_cls_method_is_true. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'root'
    var_4 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_globals_type_comment_not_none. Retrieved 11/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Assign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)
    var_10 = 'root.x'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_func_api_with_args_and_return. Retrieved 14/19 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 16/21 statements.
# Partially parsed test_func_api_with_self. Retrieved 13/18 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 19/24 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = module_1.arguments(*var_2)
    var_7 = None
    var_8 = 'root'
    var_9 = 'func'
    var_10 = False
    var_11 = var_0.func_api(var_8, var_9, var_6, var_7, has_self=var_10, cls_method=var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'int'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 'root'
    var_12 = 'func'
    var_13 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = 2
    var_10 = module_1.Constant()
    var_11 = [var_8, var_10]
    var_12 = None
    var_13 = 'root'
    var_14 = 'func'
    var_15 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = 'x'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = True
    var_12 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = 'x'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'args'
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'kwargs'
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'func'
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = []
    var_9 = None
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'y'
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'bool'
    var_14 = module_1.Load()
    var_15 = module_1.Name()
    var_16 = 'root'
    var_17 = 'func'
    var_18 = False



# Parsed testcases at query #31
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
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[int, int]'

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
    var_0 = []
    var_1 = module_0.Tuple()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = module_0.Name()
    var_2 = module_0.Call()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'list'
    var_1 = module_0.Name()
    var_2 = module_0.Call()
    var_3 = module_1.const_type(var_2)
    assert var_3 == 'list'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = module_0.BinOp()
    var_1 = module_1.const_type(var_0)
    assert var_1 == 'Any'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_func_ann_yields_empty_string_for_star_arg. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = 'root'
    var_4 = False



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 10/11 statements.
# Partially parsed test_visit_Attribute_non_typing_prefix. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
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
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 12/15 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 12/15 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/15 statements.


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
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = 'y'
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'args'
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = None
    var_8 = None
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 1
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'func'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'func'
    var_10 = True
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'func'
    var_10 = True



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = module_1.Assign()
    var_8 = var_0.globals(var_1, var_7)
    var_9 = set()



# Parsed testcases at query #36
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = 'x'
    var_5 = module_1.Name()
    var_6 = 'y'
    var_7 = module_1.Name()
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = module_1.Constant()
    var_11 = module_1.Assign()
    var_12 = [var_11]
    var_13 = var_0.class_api(var_1, var_2, var_3, var_12)
    var_14 = var_0.doc[var_2]
    var_15 = len(var_14)
    assert var_15 == 0



# Parsed testcases at query #37
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 42
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'root'
    var_8 = var_0.globals(var_7, var_6)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_func_api_with_args_and_return. Retrieved 15/29 statements.
# Partially parsed test_func_api_with_self. Retrieved 16/24 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 15/23 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 14/20 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 14/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = module_1.arguments(*var_3)
    var_8 = None
    var_9 = []
    var_10 = module_1.FunctionDef(*var_7)
    var_11 = 'root'
    var_12 = 'root.test'
    var_13 = var_10.args
    var_14 = var_10.returns
    var_15 = False
    var_16 = var_0.func_api(var_11, var_12, var_13, var_14, has_self=var_15, cls_method=var_15)
    var_17 = '| arg | return |\n|:---:|:---:|\n|  |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = 'x'
    var_4 = 1
    var_5 = 'y'
    var_6 = 2
    var_7 = []
    var_8 = []
    var_9 = 3
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test'
    var_13 = False
    var_14 = '| arg | arg | return |\n|:---:|:---:|:---:|\n| `x` | `y` | `3` |\n| `1` | `2` |  |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = 'self'
    var_4 = 'Self'
    var_5 = module_1.Load()
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test'
    var_13 = True
    var_14 = False
    var_15 = '| arg | return |\n|:---:|:---:|\n| `Self` |  |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = 'cls'
    var_4 = 'type[Self]'
    var_5 = module_1.Load()
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test'
    var_13 = True
    var_14 = '| arg | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = 'args'
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'root.test'
    var_12 = False
    var_13 = '| arg | return |\n|:---:|:---:|\n| `*args` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = 'kwargs'
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'root.test'
    var_12 = False
    var_13 = '| arg | return |\n|:---:|:---:|\n| `**kwargs` |  |\n\n'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_parser_constructor_with_toc. Retrieved 1/2 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()

def test_case_0():
    var_0 = True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 7/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'path'
    var_3 = 'sp'
    var_4 = 0
    var_5 = 'test'
    var_6 = 'test.sp'



# Parsed testcases at query #41
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = module_1.Name()
    var_4 = [var_3]
    var_5 = 123
    var_6 = module_1.Constant()
    var_7 = None
    var_8 = module_1.Assign()
    var_9 = var_0.globals(var_1, var_8)



# Parsed testcases at query #42
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = module_0.Try()
    var_5 = [var_4]
    var_6 = module_1.walk_body(var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = module_0.Try()
    var_13 = [var_12]



# Parsed testcases at query #43
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = 'Base1'
    var_4 = module_1.Name()
    var_5 = 'Base2'
    var_6 = module_1.Name()
    var_7 = [var_4, var_6]
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyEnum'
    var_3 = 'enum.Enum'
    var_4 = module_1.Name()
    var_5 = [var_4]
    var_6 = 'A'
    var_7 = module_1.Name()
    var_8 = 'int'
    var_9 = module_1.Name()
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = module_1.AnnAssign()
    var_13 = 'B'
    var_14 = module_1.Name()
    var_15 = module_1.Name()
    var_16 = 2
    var_17 = module_1.Constant()
    var_18 = module_1.AnnAssign()
    var_19 = [var_12, var_18]
    var_20 = var_0.class_api(var_1, var_2, var_5, var_19)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = module_1.AnnAssign()
    var_11 = 'another_attr'
    var_12 = module_1.Name()
    var_13 = [var_12]
    var_14 = 'value'
    var_15 = module_1.Constant()
    var_16 = 'str'
    var_17 = module_1.Assign()
    var_18 = [var_10, var_17]
    var_19 = var_0.class_api(var_1, var_2, var_3, var_18)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = []
    var_4 = 'attr1'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = module_1.AnnAssign()
    var_11 = module_1.Name()
    var_12 = [var_11]
    var_13 = module_1.Delete()
    var_14 = [var_10, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_3, var_14)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test.module.MyClass'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = module_1.Name()
    var_6 = 'int'
    var_7 = module_1.Name()
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = module_1.AnnAssign()
    var_11 = [var_10]
    var_12 = var_0.class_api(var_1, var_2, var_3, var_11)



# Parsed testcases at query #44
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'foo'
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = module_1.Tuple()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_isinstance_call_and_name_or_attribute. Retrieved 5/8 statements.


import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = module_0.Call(*var_2)
    var_4 = var_3.func



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 5/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 7/10 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 5/8 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 9/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias.value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'name'
    var_4 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'root.TypeVar'
    var_3 = "TypeVar('T')"
    var_4 = 'typing.TypeVar'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Resolver(var_0, var_5)
    var_7 = 'name'
    var_8 = module_1.Load()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_func_api_with_args_and_no_defaults. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_args_and_defaults. Retrieved 16/21 statements.
# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_self_and_not_cls_method. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 12/15 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 12/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = module_1.arguments(*var_2)
    var_7 = None
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = var_0.func_api(var_8, var_9, var_6, var_7, has_self=var_10, cls_method=var_10)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = 'b'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = 'b'
    var_5 = []
    var_6 = []
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = 2
    var_10 = module_1.Constant()
    var_11 = [var_8, var_10]
    var_12 = None
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = 'root'
    var_9 = 'name'
    var_10 = True
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'args'
    var_7 = None
    var_8 = None
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = None
    var_8 = None
    var_9 = 'root'
    var_10 = 'name'
    var_11 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = module_1.arguments(*var_2)
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.func_api(var_10, var_11, var_6, var_9, has_self=var_12, cls_method=var_12)



# Parsed testcases at query #48
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = module_1.AnnAssign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = None
    var_6 = 1
    var_7 = module_1.AnnAssign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Assign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = None
    var_7 = module_1.Assign()
    var_8 = 'root'
    var_9 = var_0.globals(var_8, var_7)

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
    var_8 = None
    var_9 = module_1.Assign()
    var_10 = 'root'
    var_11 = var_0.globals(var_10, var_9)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'x'
    var_5 = module_1.Constant()
    var_6 = 'y'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.List()
    var_10 = None
    var_11 = module_1.Assign()
    var_12 = 'root'
    var_13 = var_0.globals(var_12, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = 2
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.List()
    var_10 = None
    var_11 = module_1.Assign()
    var_12 = 'root'
    var_13 = var_0.globals(var_12, var_11)
    var_14 = set()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_func_api_with_posonlyargs_and_defaults. Retrieved 16/26 statements.
# Partially parsed test_func_api_with_vararg_and_returns. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_has_self_and_cls_method. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 16/21 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = 'd'
    var_9 = 2
    var_10 = module_1.Constant()
    var_11 = [var_10]
    var_12 = 'kwargs'
    var_13 = 'root'
    var_14 = 'func'
    var_15 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'args'
    var_4 = 'kwargs'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = 'root'
    var_9 = 'func'
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'a'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = 'func'
    var_9 = True
    var_10 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'b'
    var_6 = 'str'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 1
    var_10 = module_1.Constant()
    var_11 = [var_10]
    var_12 = 'root'
    var_13 = 'func'
    var_14 = None
    var_15 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true_sets_link_true. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_func_ann_with_has_self_and_cls_method. Retrieved 10/16 statements.
# Partially parsed test_func_ann_with_has_self_no_cls_method. Retrieved 11/17 statements.
# Partially parsed test_func_ann_without_has_self. Retrieved 9/14 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 10/16 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 6/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'return'
    var_8 = 'module'
    var_9 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'return'
    var_8 = 'module'
    var_9 = True
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'return'
    var_6 = None
    var_7 = 'module'
    var_8 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = '*'
    var_6 = None
    var_7 = 'return'
    var_8 = 'module'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'return'
    var_4 = 'module'
    var_5 = False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_globals_with_annassign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_ignores_complex_assign. Retrieved 6/13 statements.
# Partially parsed test_globals_ignores_non_constant_all. Retrieved 7/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 'hello'
    var_3 = module_1.Constant()
    var_4 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'foo'
    var_3 = module_1.Constant()
    var_4 = 'bar'
    var_5 = module_1.Constant()
    var_6 = [var_3, var_5]
    var_7 = module_1.List()
    var_8 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = module_1.Constant()
    var_5 = 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'some_var'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'root'
    var_6 = set()



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = 'a'
    var_4 = None
    var_5 = module_1.arg()
    var_6 = 'b'
    var_7 = module_1.arg()
    var_8 = [var_5, var_7]
    var_9 = 'c'
    var_10 = module_1.arg()
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = module_1.arguments(*var_11)
    var_16 = None
    var_17 = False
    var_18 = False
    var_19 = var_0.func_api(var_1, var_2, var_15, var_16, has_self=var_17, cls_method=var_18)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_isinstance_d_not_name. Retrieved 6/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'class Test:\n    del x[0]'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test.Test'
    var_5 = var_0.doc[var_4]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 7/8 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_globals_with_annassign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 6/11 statements.
# Partially parsed test_globals_with_assign_no_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 7/12 statements.
# Partially parsed test_globals_with_non_uppercase. Retrieved 5/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 42
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'int'
    var_5 = 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_var'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'public_func'
    var_3 = module_1.Constant()
    var_4 = [var_3]
    var_5 = module_1.List()
    var_6 = 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_upper'
    var_2 = 42
    var_3 = module_1.Constant()
    var_4 = 'test_module'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_imports_with_import_node. Retrieved 4/8 statements.
# Partially parsed test_imports_with_import_node_and_asname. Retrieved 4/8 statements.
# Partially parsed test_imports_with_import_from_node. Retrieved 6/10 statements.
# Partially parsed test_imports_with_import_from_node_and_asname. Retrieved 6/10 statements.
# Partially parsed test_imports_with_import_from_node_and_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = None
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = 'path_join'
    var_5 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'sibling'
    var_3 = 'func'
    var_4 = None
    var_5 = 1



# Parsed testcases at query #62
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_compile_empty_parser. Retrieved 2/4 statements.
# Partially parsed test_compile_with_basic_module. Retrieved 4/6 statements.
# Partially parsed test_compile_with_non_public_items. Retrieved 4/6 statements.
# Partially parsed test_compile_with_magic_method. Retrieved 4/6 statements.
# Partially parsed test_compile_with_toc_disabled. Retrieved 5/7 statements.
# Partially parsed test_compile_with_link_disabled. Retrieved 6/8 statements.
# Partially parsed test_compile_with_constants. Retrieved 4/6 statements.
# Partially parsed test_compile_with_class_members. Retrieved 4/6 statements.
# Partially parsed test_compile_with_inheritance. Retrieved 4/6 statements.
# Partially parsed test_compile_with_decorators. Retrieved 4/6 statements.
# Partially parsed test_compile_with_nested_classes. Retrieved 4/6 statements.
# Partially parsed test_compile_with_enum. Retrieved 4/6 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = 'def func():\n    pass'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\ndef _private_func():\n    pass\n\ndef public_func():\n    pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    def __init__(self):\n        pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_module'
    var_3 = 'def func():\n    pass'
    var_4 = module_0.parse(var_2, var_3)

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True
    var_3 = 'test_module'
    var_4 = 'def func():\n    pass'
    var_5 = module_0.parse(var_3, var_4)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nMAX_SIZE = 100\ndef func():\n    pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nclass MyClass:\n    def __init__(self):\n        self.value: int = 0\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nclass Parent:\n    pass\n\nclass Child(Parent):\n    pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\ndef decorator(func):\n    return func\n\n@decorator\ndef func():\n    pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nclass Outer:\n    class Inner:\n        pass\n'
    var_3 = module_0.parse(var_1, var_2)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_module'
    var_2 = '\nfrom enum import Enum\n\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n'
    var_3 = module_0.parse(var_1, var_2)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 9/15 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 9/13 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 8/11 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 8/11 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_self. Retrieved 10/15 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 9/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'root.func'
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'root.func'
    var_8 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'args'
    var_3 = None
    var_4 = None
    var_5 = 'root'
    var_6 = 'root.func'
    var_7 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'kwargs'
    var_3 = None
    var_4 = None
    var_5 = 'root'
    var_6 = 'root.func'
    var_7 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = None
    var_8 = 'root'
    var_9 = 'root.func'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'root.Class.func'
    var_8 = True
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = 'x'
    var_4 = []
    var_5 = None
    var_6 = 'root'
    var_7 = 'root.Class.func'
    var_8 = True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_globals_with_non_constant. Retrieved 10/11 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'test_var'
    var_1 = module_0.Name()
    var_2 = 'int'
    var_3 = module_0.Name()
    var_4 = 42
    var_5 = module_0.Constant()
    var_6 = module_0.AnnAssign()
    var_7 = module_1.Parser()
    var_8 = 'test.module'
    var_9 = var_7.globals(var_8, var_6)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'test_var'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 42
    var_4 = module_0.Constant()
    var_5 = module_0.Assign()
    var_6 = module_1.Parser()
    var_7 = 'test.module'
    var_8 = var_6.globals(var_7, var_5)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'test_var'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 42
    var_4 = module_0.Constant()
    var_5 = 'int'
    var_6 = module_0.Assign()
    var_7 = module_1.Parser()
    var_8 = 'test.module'
    var_9 = var_7.globals(var_8, var_6)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = '__all__'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 'test_var'
    var_4 = module_0.Constant()
    var_5 = [var_4]
    var_6 = module_0.List()
    var_7 = module_0.Assign()
    var_8 = module_1.Parser()
    var_9 = 'test.module'
    var_10 = var_8.globals(var_9, var_7)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'test_var'
    var_1 = module_0.Name()
    var_2 = [var_1]
    var_3 = 'some_var'
    var_4 = module_0.Name()
    var_5 = module_0.Assign()
    var_6 = module_1.Parser()
    var_7 = 'test.module'
    var_8 = var_6.globals(var_7, var_5)
    var_9 = 'test.module.test_var'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_func_ann_with_self_and_cls_method. Retrieved 9/14 statements.
# Partially parsed test_func_ann_with_self_no_cls_method. Retrieved 10/15 statements.
# Partially parsed test_func_ann_without_self. Retrieved 11/16 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 13/19 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 6/11 statements.
# Partially parsed test_func_ann_with_self_type_annotation. Retrieved 12/17 statements.
# Partially parsed test_func_ann_with_cls_method_and_self_type. Retrieved 11/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'root'
    var_8 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = 'x'
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'root'
    var_8 = True
    var_9 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'y'
    var_6 = 'str'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'root'
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = '*'
    var_6 = None
    var_7 = 'y'
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 'root'
    var_12 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'MyClass'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'x'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'root'
    var_10 = True
    var_11 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'type[MyClass]'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'x'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'root'
    var_10 = True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_compile_skips_magic_names. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = '## Module `{}`\n\n'
    var_3 = 0
    var_4 = var_0.compile()
    assert var_4 == ''



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_visit_Name_self_ty_match. Retrieved 7/8 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'Test'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = var_3.visit_Name(var_5)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test__attr_returns_none_for_nonexistent_nested_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent_attr'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 9/14 statements.
# Partially parsed test_globals_with_non_public_constant. Retrieved 9/13 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 6/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = 3.14
    var_3 = module_1.Constant()
    var_4 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = 'func1'
    var_3 = module_1.Constant()
    var_4 = 'func2'
    var_5 = module_1.Constant()
    var_6 = [var_3, var_5]
    var_7 = module_1.List()
    var_8 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private'
    var_2 = 'str'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'secret'
    var_6 = module_1.Constant()
    var_7 = 1
    var_8 = 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = True
    var_3 = module_1.Constant()
    var_4 = 'bool'
    var_5 = 'module'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_const_type_constant. Retrieved 3/9 statements.
# Partially parsed test_const_type_tuple. Retrieved 3/13 statements.
# Partially parsed test_const_type_list. Retrieved 3/13 statements.
# Partially parsed test_const_type_set. Retrieved 3/13 statements.
# Partially parsed test_const_type_dict. Retrieved 4/26 statements.
# Partially parsed test_const_type_call. Retrieved 16/31 statements.
# Partially parsed test_const_type_any. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 5
    var_1 = 3.14
    var_2 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'

def test_case_0():
    var_0 = 'bool'
    var_1 = None
    var_2 = []
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = []
    var_7 = 'float'
    var_8 = []
    var_9 = []
    var_10 = 'complex'
    var_11 = []
    var_12 = []
    var_13 = 'str'
    var_14 = []
    var_15 = []

def test_case_0():
    var_0 = 'unknown'
    var_1 = None
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_imports_with_regular_import. Retrieved 4/8 statements.
# Partially parsed test_imports_with_asname. Retrieved 4/8 statements.
# Partially parsed test_imports_with_from_import. Retrieved 6/10 statements.
# Partially parsed test_imports_with_from_import_and_asname. Retrieved 6/10 statements.
# Partially parsed test_imports_with_relative_import. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = None
    var_3 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'system'
    var_3 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'exit'
    var_3 = None
    var_4 = 0
    var_5 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'exit'
    var_3 = 'quit'
    var_4 = 0
    var_5 = 'pkg'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'subpkg'
    var_2 = 'func'
    var_3 = None
    var_4 = 1
    var_5 = 'pkg.subpkg'



# Parsed testcases at query #5
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nimport os\nfrom sys import path\nx = 1\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nCONSTANT = 42\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nclass Outer:\n    class Inner:\n        pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nif True:\n    x = 1\nelse:\n    y = 2\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 8/12 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 8/12 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 8/12 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 8/12 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 8/12 statements.
# Partially parsed test_is_public_with_empty_all. Retrieved 7/11 statements.
# Partially parsed test_is_public_with_const_in_all. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name'
    var_5 = 'root.other'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root._private'
    var_5 = 'root.other'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = '__init__'
    var_3 = {var_2}
    var_4 = 'root.__init__'
    var_5 = 'root.other'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name.nested'
    var_5 = 'root.other'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name._nested'
    var_5 = 'root.other'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.public_name'
    var_4 = 'root.other'
    var_5 = ''
    var_6 = var_0.is_public(var_3)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'CONST_NAME'
    var_3 = {var_2}
    var_4 = 'root.CONST_NAME'
    var_5 = 'int'
    var_6 = var_0.is_public(var_4)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__e_type_none_element. Retrieved 2/3 statements.
# Partially parsed test__e_type_mixed_constant_types. Retrieved 2/6 statements.
# Partially parsed test__e_type_single_constant_type. Retrieved 2/6 statements.
# Partially parsed test__e_type_multiple_sequences. Retrieved 2/7 statements.
# Partially parsed test__e_type_non_constant_element. Retrieved 2/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0._e_type()
    assert var_0 == ''

def test_case_0():
    var_0 = None
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 5/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 8/11 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 6/9 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 8/11 statements.


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
    var_2 = 'alias.value'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
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

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = module_1.Load()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.


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
    var_0 = False
    var_1 = 3
    var_2 = True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__defaults_with_values. Retrieved 2/7 statements.
# Partially parsed test__defaults_with_and_without_values. Retrieved 2/6 statements.
# Partially parsed test__defaults_with_ampersand. Retrieved 1/5 statements.
# Partially parsed test__defaults_with_pipe. Retrieved 1/5 statements.
# Partially parsed test__defaults_with_empty_string. Retrieved 1/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'

def test_case_0():
    var_0 = None
    var_1 = 'x'

def test_case_0():
    var_0 = 'a & b'

def test_case_0():
    var_0 = 'a | b'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_func_api_with_posonlyargs_and_defaults. Retrieved 13/20 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 17/22 statements.
# Partially parsed test_func_api_with_returns. Retrieved 13/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = 'b'
    var_6 = []
    var_7 = 'args'
    var_8 = 'kwargs'
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = 'a'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'func'
    var_10 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = 'int'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'b'
    var_7 = 'str'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = None
    var_14 = 'root'
    var_15 = 'func'
    var_16 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'int'
    var_8 = module_1.Load()
    var_9 = module_1.Name()
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_func_api_with_single_arg_and_return. Retrieved 14/18 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 14/19 statements.
# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 13/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = module_1.arguments(*var_2)
    var_7 = 'root'
    var_8 = 'func'
    var_9 = None
    var_10 = False
    var_11 = var_0.func_api(var_7, var_8, var_6, var_9, has_self=var_10, cls_method=var_10)
    var_12 = '| arg | return |\n|:---:|:---:|\n|  |  |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'func'
    var_9 = 'int'
    var_10 = module_1.Load()
    var_11 = module_1.Name()
    var_12 = False
    var_13 = '| arg | return |\n|:---:|:---:|\n| `x` | `int` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'
    var_5 = []
    var_6 = []
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = [var_8]
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False
    var_13 = '| arg | return |\n|:---:|:---:|\n| `x` | `y` | `return` |\n|  | `1` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'func'
    var_9 = True
    var_10 = '| arg | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'x'
    var_4 = None
    var_5 = 2
    var_6 = module_1.Constant()
    var_7 = [var_6]
    var_8 = []
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False
    var_12 = '| arg | return |\n|:---:|:---:|\n| `x` |  |\n| `2` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'args'
    var_7 = None
    var_8 = 'kwargs'
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False
    var_12 = '| arg | return |\n|:---:|:---:|\n| `*args` | `**kwargs` |  |\n\n'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'alias'
    var_3 = 'root'
    var_4 = 'root.alias'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_name'
    var_2 = 'some_value'
    var_3 = 'ANY'



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = module_1.AnnAssign()
    var_8 = 'module'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'hello'
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'module'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'foo'
    var_5 = module_1.Constant()
    var_6 = 'bar'
    var_7 = module_1.Constant()
    var_8 = [var_5, var_7]
    var_9 = module_1.List()
    var_10 = module_1.Assign()
    var_11 = 'module'
    var_12 = var_0.globals(var_11, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'some_var'
    var_5 = module_1.Name()
    var_6 = module_1.Assign()
    var_7 = 'module'
    var_8 = var_0.globals(var_7, var_6)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_func_ann_with_self_and_annotation. Retrieved 12/17 statements.
# Partially parsed test_func_ann_with_cls_method. Retrieved 11/16 statements.
# Partially parsed test_func_ann_without_annotation. Retrieved 9/14 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 13/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'MyClass'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'x'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'root'
    var_10 = True
    var_11 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'MyClass'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'x'
    var_6 = 'int'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = 'root'
    var_10 = True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = 'str'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = 'root'
    var_8 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = '*'
    var_6 = None
    var_7 = 'y'
    var_8 = 'str'
    var_9 = module_1.Load()
    var_10 = module_1.Name()
    var_11 = 'root'
    var_12 = False



# Parsed testcases at query #18
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
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_class_api_with_enum. Retrieved 15/23 statements.
# Partially parsed test_class_api_with_members. Retrieved 13/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 9/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = []
    var_8 = var_0.class_api(var_1, var_2, var_6, var_7)

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'Enum'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 'VALUE1'
    var_8 = 1
    var_9 = module_1.Constant()
    var_10 = module_1.Constant()
    var_11 = 'VALUE2'
    var_12 = 2
    var_13 = module_1.Constant()
    var_14 = module_1.Constant()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'member1'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = None
    var_9 = 'member2'
    var_10 = 'str'
    var_11 = module_1.Load()
    var_12 = module_1.Name()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'member1'
    var_5 = 'int'
    var_6 = module_1.Load()
    var_7 = module_1.Name()
    var_8 = None



# Parsed testcases at query #20
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
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[int, int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Constant()
    var_2 = 'b'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'list[str, str]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1.1
    var_1 = module_0.Constant()
    var_2 = 2.2
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Set()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'set[float, float]'

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
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'a'
    var_3 = module_0.Constant()
    var_4 = [var_1, var_3]
    var_5 = module_0.Tuple()
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Tuple()
    var_2 = module_1.const_type(var_1)
    assert var_2 == 'tuple[]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = module_0.List()
    var_5 = module_1.const_type(var_4)
    assert var_5 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = module_0.Call(*var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = module_0.Call(*var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'int'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown'
    var_1 = module_0.Name()
    var_2 = []
    var_3 = module_0.Call(*var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'Any'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Constant()
    var_2 = 'x'
    var_3 = module_0.Name()
    var_4 = [var_1, var_3]
    var_5 = module_0.List()
    var_6 = module_1.const_type(var_5)
    assert var_6 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'obj'
    var_1 = module_0.Name()
    var_2 = 'method'
    var_3 = module_0.Attribute()
    var_4 = []
    var_5 = module_0.Call(*var_4)
    var_6 = module_1.const_type(var_5)
    assert var_6 == 'Any'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_class_api. Retrieved 5/6 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = '\nclass TestClass(BaseClass):\n    """Test class docstring."""\n    attr1: int\n    attr2: str = "default"\n    _private_attr: bool = True\n    __magic_attr__: float = 1.0\n\n    def __init__(self):\n        pass\n'
    var_4 = module_0.parse(var_2, var_3)



# Parsed testcases at query #22
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
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test_attr_missing_intermediate_attribute. Retrieved 2/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'x'

def test_case_0():
    var_0 = 20
    var_1 = 'inner.y'

def test_case_0():
    var_0 = 10
    var_1 = 'z'

def test_case_0():
    var_0 = 20
    var_1 = 'inner.z'

def test_case_0():
    var_0 = 10
    var_1 = 'inner.y'

def test_case_0():
    var_0 = 10
    var_1 = ''



# Parsed testcases at query #24
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = module_1.Name()
    var_3 = 'int'
    var_4 = module_1.Name()
    var_5 = 5
    var_6 = module_1.Constant()
    var_7 = module_1.AnnAssign()
    var_8 = 'module'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 5
    var_5 = module_1.Constant()
    var_6 = module_1.Assign()
    var_7 = 'module'
    var_8 = var_0.globals(var_7, var_6)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 5
    var_5 = module_1.Constant()
    var_6 = 'int'
    var_7 = module_1.Assign()
    var_8 = 'module'
    var_9 = var_0.globals(var_8, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'func'
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = module_1.List()
    var_8 = module_1.Assign()
    var_9 = 'module'
    var_10 = var_0.globals(var_9, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var1'
    var_2 = module_1.Name()
    var_3 = 'var2'
    var_4 = module_1.Name()
    var_5 = [var_2, var_4]
    var_6 = 5
    var_7 = module_1.Constant()
    var_8 = module_1.Assign()
    var_9 = 'module'
    var_10 = var_0.globals(var_9, var_8)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_compile_empty. Retrieved 2/4 statements.
# Partially parsed test_compile_with_toc. Retrieved 5/11 statements.
# Partially parsed test_compile_with_docstring. Retrieved 6/13 statements.
# Partially parsed test_compile_with_const. Retrieved 6/13 statements.
# Partially parsed test_compile_with_magic. Retrieved 6/12 statements.
# Partially parsed test_compile_with_link. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True
    var_1 = '#'
    var_2 = 2
    var_3 = var_1 * var_2
    var_4 = ' Module `{}`'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = '#'
    var_3 = 2
    var_4 = var_2 * var_3
    var_5 = ' Module `{}`'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = '#'
    var_3 = 2
    var_4 = var_2 * var_3
    var_5 = ' Module `{}`'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = '#'
    var_3 = 3
    var_4 = var_2 * var_3
    var_5 = ' `{}`()\n\n'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = '#'
    var_3 = 2
    var_4 = var_2 * var_3
    var_5 = ' Module `{}`\n<a id="{}"></a>\n\n'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_imports_with_Import_node. Retrieved 4/8 statements.
# Partially parsed test_imports_with_Import_node_and_asname. Retrieved 4/8 statements.
# Partially parsed test_imports_with_ImportFrom_node_with_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_ImportFrom_node_without_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = None

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'operating_system'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_magic_predicate. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = '__add__'
    var_3 = ''
    var_4 = 0
    var_5 = var_0.compile()
    assert var_5 == ''



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_api_function. Retrieved 11/12 statements.
# Partially parsed test_api_async_function. Retrieved 11/12 statements.
# Partially parsed test_api_class. Retrieved 10/11 statements.
# Partially parsed test_api_with_prefix. Retrieved 12/13 statements.
# Partially parsed test_api_with_docstring. Retrieved 12/13 statements.
# Partially parsed test_api_class_with_enum. Retrieved 10/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = module_1.arguments()
    var_4 = None
    var_5 = []
    var_6 = module_1.FunctionDef(*var_3)
    var_7 = var_0.api(var_1, var_6)
    var_8 = 'test_module.test_func'
    var_9 = var_0.doc[var_8]
    var_10 = '### test_func()'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'async_func'
    var_3 = module_1.arguments()
    var_4 = None
    var_5 = []
    var_6 = module_1.AsyncFunctionDef(*var_3)
    var_7 = var_0.api(var_1, var_6)
    var_8 = 'test_module.async_func'
    var_9 = var_0.doc[var_8]
    var_10 = '### async async_func()'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = module_1.ClassDef()
    var_6 = var_0.api(var_1, var_5)
    var_7 = 'test_module.TestClass'
    var_8 = var_0.doc[var_7]
    var_9 = '### class TestClass'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'method'
    var_3 = module_1.arguments()
    var_4 = None
    var_5 = []
    var_6 = module_1.FunctionDef(*var_3)
    var_7 = 'TestClass'
    var_8 = var_0.api(var_1, var_6, prefix=var_7)
    var_9 = 'test_module.TestClass.method'
    var_10 = var_0.doc[var_9]
    var_11 = '#### method()'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'staticmethod'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'decorated_func'
    var_6 = module_1.arguments()
    var_7 = None
    var_8 = []
    var_9 = [var_4]
    var_10 = module_1.FunctionDef(*var_6)
    var_11 = var_0.api(var_1, var_10)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'doc_func'
    var_3 = module_1.arguments()
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = module_1.FunctionDef(*var_3)
    var_8 = 'This is a docstring'
    var_9 = module_1.Constant()
    var_10 = module_1.Expr()
    var_11 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'BaseClass'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'DerivedClass'
    var_6 = [var_4]
    var_7 = []
    var_8 = module_1.ClassDef()
    var_9 = var_0.api(var_1, var_8)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'member'
    var_3 = module_1.Name()
    var_4 = 'int'
    var_5 = module_1.Load()
    var_6 = module_1.Name()
    var_7 = None
    var_8 = module_1.AnnAssign()
    var_9 = 'MemberClass'
    var_10 = []
    var_11 = [var_8]
    var_12 = module_1.ClassDef()
    var_13 = var_0.api(var_1, var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'enum.Enum'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'VALUE1'
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 'EnumClass'
    var_9 = [var_4]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 7/11 statements.
# Partially parsed test_visit_Attribute_keeps_non_typing_prefix. Retrieved 7/11 statements.
# Partially parsed test_visit_Attribute_returns_node_if_value_not_Name. Retrieved 6/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not_a_name'
    var_4 = 'List'
    var_5 = module_1.Load()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 1/6 statements.
# Partially parsed test_attr_multi_level. Retrieved 1/9 statements.
# Partially parsed test_attr_nonexistent. Retrieved 1/5 statements.
# Partially parsed test_attr_partial_nonexistent. Retrieved 1/8 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'

def test_case_0():
    var_0 = 'nested.value'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'nested.nonexistent'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n'
    var_3 = var_0.parse(var_1, var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_visit_constant_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_invalid_syntax. Retrieved 4/6 statements.
# Partially parsed test_visit_constant_valid_name. Retrieved 6/9 statements.
# Partially parsed test_visit_constant_valid_name_with_self. Retrieved 4/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax !@#'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_api_with_decorators. Retrieved 18/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = []
    var_11 = module_1.FunctionDef(*var_8)
    var_12 = var_0.api(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = []
    var_11 = module_1.AsyncFunctionDef(*var_8)
    var_12 = var_0.api(var_1, var_11)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = module_1.ClassDef()
    var_8 = var_0.api(var_1, var_7)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'nested_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = []
    var_11 = module_1.FunctionDef(*var_8)
    var_12 = 'OuterClass'
    var_13 = var_0.api(var_1, var_11, prefix=var_12)

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'decorator'
    var_3 = module_1.Load()
    var_4 = module_1.Name()
    var_5 = 'decorated_func'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = module_1.arguments(*var_7)
    var_12 = None
    var_13 = [var_4]
    var_14 = module_1.FunctionDef(*var_11)
    var_15 = var_0.api(var_1, var_14)
    var_16 = var_0.doc[var_5]
    var_17 = '### decorated_func()\n\n*Full name:* `test.decorated_func`\n<a id="test-decorated_func"></a>\n\n| Decorators |\n|:-----------:|\n| @decorator |'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test'
    var_2 = 'doc_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.arguments(*var_4)
    var_9 = None
    var_10 = []
    var_11 = 'This is a docstring'
    var_12 = module_1.Constant()
    var_13 = module_1.Expr()
    var_14 = [var_13]
    var_15 = module_1.FunctionDef(*var_8)
    var_16 = var_0.api(var_1, var_15)



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 123
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = module_1.List()
    var_8 = module_1.Assign()
    var_9 = 'root'
    var_10 = var_0.globals(var_9, var_8)



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = module_1.Constant()
    var_5 = var_2.visit_Constant(var_4)



# Parsed testcases at query #36
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = 'a'
    var_4 = None
    var_5 = module_1.arg()
    var_6 = [var_5]
    var_7 = 'b'
    var_8 = module_1.arg()
    var_9 = [var_8]
    var_10 = 1
    var_11 = module_1.Constant()
    var_12 = [var_11]
    var_13 = 'args'
    var_14 = module_1.arg()
    var_15 = []
    var_16 = []
    var_17 = module_1.arguments(*var_9)
    var_18 = None
    var_19 = False
    var_20 = False
    var_21 = var_0.func_api(var_1, var_2, var_17, var_18, has_self=var_19, cls_method=var_20)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 7/11 statements.
# Partially parsed test_visit_Attribute_non_typing_prefix. Retrieved 7/11 statements.
# Partially parsed test_visit_Attribute_non_name_value. Retrieved 6/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = 'List'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = module_1.Load()
    var_5 = 'attr'
    var_6 = module_1.Load()

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'not a name'
    var_4 = 'attr'
    var_5 = module_1.Load()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_self. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_classmethod. Retrieved 10/14 statements.


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
    var_8 = 'root'
    var_9 = 'test_func'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'a'
    var_5 = None
    var_6 = 'b'
    var_7 = []
    var_8 = 'root'
    var_9 = 'test_func'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'args'
    var_8 = 'root'
    var_9 = 'test_func'
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = None
    var_8 = 'root'
    var_9 = 'test_func'
    var_10 = False

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'test_func'
    var_11 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'test_func'
    var_9 = True
    var_10 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'root'
    var_8 = 'test_func'
    var_9 = True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_func_ann_with_star_arg. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '*'
    var_2 = None
    var_3 = 'root'
    var_4 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 16/20 statements.
# Partially parsed test_class_api_with_enums. Retrieved 20/28 statements.
# Partially parsed test_class_api_with_members. Retrieved 19/27 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/20 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'BaseClass'
    var_3 = module_0.Load()
    var_4 = module_0.Name()
    var_5 = [var_4]
    var_6 = []
    var_7 = 'root.module'
    var_8 = 'root.module.ClassName'
    var_9 = '#'
    var_10 = 3
    var_11 = var_9 * var_10
    var_12 = ' class ClassName\n\n*Full name:* `{}`\n\n'
    var_13 = var_11 + var_12
    var_14 = 'Bases'
    var_15 = [var_2]

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'enum.Enum'
    var_3 = module_0.Load()
    var_4 = module_0.Name()
    var_5 = [var_4]
    var_6 = 'RED'
    var_7 = 'int'
    var_8 = module_0.Load()
    var_9 = module_0.Name()
    var_10 = None
    var_11 = 'root.module'
    var_12 = 'root.module.Color'
    var_13 = '#'
    var_14 = 3
    var_15 = var_13 * var_14
    var_16 = ' class Color\n\n*Full name:* `{}`\n\n'
    var_17 = var_15 + var_16
    var_18 = 'Enums'
    var_19 = [var_6]

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'attr1'
    var_4 = 'int'
    var_5 = module_0.Load()
    var_6 = module_0.Name()
    var_7 = None
    var_8 = 'root.module'
    var_9 = 'root.module.ClassName'
    var_10 = '#'
    var_11 = 3
    var_12 = var_10 * var_11
    var_13 = ' class ClassName\n\n*Full name:* `{}`\n\n'
    var_14 = var_12 + var_13
    var_15 = 'Members'
    var_16 = 'Type'
    var_17 = (var_3, var_4)
    var_18 = [var_17]

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'attr1'
    var_4 = 'int'
    var_5 = module_0.Load()
    var_6 = module_0.Name()
    var_7 = None
    var_8 = 'root.module'
    var_9 = 'root.module.ClassName'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_const_type_with_constant. Retrieved 4/12 statements.
# Partially parsed test_const_type_with_tuple. Retrieved 4/16 statements.
# Partially parsed test_const_type_with_list. Retrieved 4/16 statements.
# Partially parsed test_const_type_with_set. Retrieved 4/16 statements.
# Partially parsed test_const_type_with_dict. Retrieved 6/22 statements.
# Partially parsed test_const_type_with_call. Retrieved 13/32 statements.


def test_case_0():
    var_0 = 5
    var_1 = 3.14
    var_2 = 'hello'
    var_3 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'int'
    var_3 = []
    var_4 = 'float'
    var_5 = []
    var_6 = 'complex'
    var_7 = []
    var_8 = 'str'
    var_9 = []
    var_10 = 'x'
    var_11 = 'y'
    var_12 = []



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 10/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = module_1.Load()
    var_5 = module_1.Name()
    var_6 = 'List'
    var_7 = module_1.Load()
    var_8 = module_1.Attribute()
    var_9 = var_2.visit_Attribute(var_8)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_class_api. Retrieved 10/12 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = '\nclass TestClass(BaseClass):\n    """Test class docstring."""\n    x: int\n    y = 1\n    def method(self):\n        pass\n'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'test_module.TestClass'
    var_6 = 'BaseClass'
    var_7 = module_0.Name()
    var_8 = [var_7]
    var_9 = []



# Parsed testcases at query #44
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = module_1.Name()
    var_3 = [var_2]
    var_4 = 'foo'
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = module_1.Load()
    var_8 = module_1.Tuple()
    var_9 = module_1.Assign()
    var_10 = 'root'
    var_11 = var_0.globals(var_10, var_9)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 11/19 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 11/17 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 11/15 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 14/20 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 12/18 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 13/19 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'a'
    var_3 = None
    var_4 = 'b'
    var_5 = 'c'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'root.func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'a'
    var_4 = None
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'args'
    var_9 = 'root'
    var_10 = 'root.func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'a'
    var_6 = None
    var_7 = 'b'
    var_8 = []
    var_9 = 'root'
    var_10 = 'root.func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'kwargs'
    var_8 = None
    var_9 = 'root'
    var_10 = 'root.func'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'a'
    var_4 = None
    var_5 = 'b'
    var_6 = module_0.Constant()
    var_7 = 2
    var_8 = module_0.Constant()
    var_9 = [var_6, var_8]
    var_10 = []
    var_11 = []
    var_12 = 'root'
    var_13 = 'root.func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'self'
    var_4 = None
    var_5 = 'a'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'root.func'
    var_11 = True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'cls'
    var_4 = None
    var_5 = 'a'
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'root.func'
    var_11 = True
    var_12 = True



# Parsed testcases at query #46
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = 'test_attr'
    var_5 = module_1.Name()
    var_6 = [var_5]
    var_7 = 42
    var_8 = module_1.Constant()
    var_9 = None
    var_10 = module_1.Assign()
    var_11 = [var_10]
    var_12 = var_0.class_api(var_1, var_2, var_3, var_11)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_func_api_with_kwonlyargs_and_no_vararg. Retrieved 15/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = []
    var_4 = []
    var_5 = 'kwarg1'
    var_6 = None
    var_7 = 'kwarg2'
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = False
    var_12 = False
    var_13 = var_0.doc[var_2]
    var_14 = '\n\n| * | kwarg1 | kwarg2 | return |\n|---|---|---|---|\n|  | ANY | ANY | ANY |\n'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_docstring. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'Module `pkg.module`'
    var_4 = 'func()'
    var_5 = None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 13/19 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 14/17 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 14/17 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 15/19 statements.
# Partially parsed test_func_api_with_self. Retrieved 13/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 13/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = 'y'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'root'
    var_11 = 'root.func'
    var_12 = '| x | y | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = None
    var_6 = []
    var_7 = None
    var_8 = False
    var_9 = False
    var_10 = 'root'
    var_11 = 'root.func'
    var_12 = '| * | x | return |\n|:---:|:---:|:---:|\n|  | `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'args'
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = 'root'
    var_12 = 'root.func'
    var_13 = '| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = None
    var_8 = None
    var_9 = False
    var_10 = False
    var_11 = 'root'
    var_12 = 'root.func'
    var_13 = '| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'x'
    var_3 = None
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = False
    var_11 = False
    var_12 = 'root'
    var_13 = 'root.func'
    var_14 = '| x | return |\n|:---:|:---:|\n| `Any` | `Any` |\n| `1` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = True
    var_9 = False
    var_10 = 'root'
    var_11 = 'root.func'
    var_12 = '| Self | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = None
    var_8 = False
    var_9 = True
    var_10 = 'root'
    var_11 = 'root.func'
    var_12 = '| type[Self] | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'



# Parsed testcases at query #50
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._local'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public.__magic__._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public1.public2.public3'
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
    var_0 = '__public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'public__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False



