####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_imports_with_import_node. Retrieved 6/11 statements.
# Partially parsed test_imports_with_import_from_node_no_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_import_from_node_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'sys'
    var_3 = None
    var_4 = 'os'
    var_5 = 'operating_system'
    var_6 = var_0.alias['test.module.sys']
    assert var_6 == 'sys'
    var_7 = var_0.alias['test.module.operating_system']
    assert var_7 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = None
    var_5 = 0
    var_6 = var_0.alias['test.module.path']
    assert var_6 == 'sys.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'sibling'
    var_3 = 'helper'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['test.module.submodule.helper']
    assert var_6 == 'test.module.sibling.helper'



# Parsed testcases at query #2
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
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a&b'
    var_1 = 'c&d'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['<code>a&b</code>', '<code>c&d</code>'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a|b'
    var_1 = 'c|d'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['`a&#124;b`', '`c&#124;d`'])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'b&c'
    var_3 = 'd|e'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._defaults(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == ['`a`', ' ', '<code>b&c</code>', '`d&#124;e`'])
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__attr_basic. Retrieved 2/4 statements.
# Partially parsed test__attr_nested. Retrieved 2/6 statements.
# Partially parsed test__attr_nonexistent. Retrieved 2/4 statements.
# Partially parsed test__attr_nonexistent_nested. Retrieved 2/6 statements.
# Partially parsed test__attr_chain_break. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'x'

def test_case_0():
    var_0 = 2
    var_1 = 'b.y'

def test_case_0():
    var_0 = 1
    var_1 = 'z'

def test_case_0():
    var_0 = 2
    var_1 = 'b.z'

def test_case_0():
    var_0 = 2
    var_1 = 'b.z.w'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/9 statements.
# Partially parsed test_globals_with_all. Retrieved 5/13 statements.
# Partially parsed test_globals_with_non_uppercase. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'int'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.alias['root.x']
    assert var_8 == '1'
    var_9 = var_0.const['root.x']
    assert var_9 == 'int'
    var_10 = var_0.root['root.x']
    assert var_10 == 'root'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 2.5
    var_4 = []
    var_5 = 'root'
    var_6 = var_0.alias['root.y']
    assert var_6 == '2.5'
    var_7 = var_0.const['root.y']
    assert var_7 == 'float'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'foo'
    var_4 = []
    var_5 = 'bar'
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.imp['root']
    var_9 = bool(var_0.imp['root'] == {'root.foo', 'root.bar'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'z'
    var_2 = []
    var_3 = 'hello'
    var_4 = []
    var_5 = 'root'
    var_6 = var_0.alias['root.z']
    assert var_6 == "'hello'"
    var_7 = 'root.z'
    var_8 = bool('root.z' not in var_0.const)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 4/7 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_listed_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_parent_listed_in_all. Retrieved 4/6 statements.
# Partially parsed test_is_public_with_empty_all. Retrieved 3/6 statements.
# Partially parsed test_is_public_with_non_public_family. Retrieved 3/5 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 3/6 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'public_name'
    var_2 = 'module.public_name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'public_name'
    var_2 = 'module._private_name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'listed_name'
    var_2 = 'module.listed_name'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent_name'
    var_2 = 'module.parent_name.child'
    var_3 = var_0.is_public(var_2)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.public_name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module._private_name'
    var_2 = var_0.is_public(var_1)
    assert var_2 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module.__magic__'
    var_2 = var_0.is_public(var_1)
    assert var_2 is True



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_class_api_basic. Retrieved 10/23 statements.
# Partially parsed test_class_api_with_members. Retrieved 12/28 statements.
# Partially parsed test_class_api_with_enum. Retrieved 13/37 statements.
# Partially parsed test_class_api_with_deleted_member. Retrieved 12/32 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = 'Base'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'test'
    var_9 = 'test.TestClass'
    var_10 = '# test\n\n*Full name:* `test.TestClass`\n\n| Bases |\n|:---:|\n| `Base` |\n\n'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'member'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'TestClass'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test'
    var_11 = 'test.TestClass'
    var_12 = '# test\n\n*Full name:* `test.TestClass`\n\n| Members | Type |\n|:---:|:---:|\n| `member` | `int` |\n\n'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'ENUM1'
    var_3 = []
    var_4 = None
    var_5 = 'ENUM2'
    var_6 = 2
    var_7 = []
    var_8 = 'TestEnum'
    var_9 = 'enum.Enum'
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'test'
    var_14 = 'test.TestEnum'
    var_15 = '# test\n\n*Full name:* `test.TestEnum`\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n\n| Enums |\n|:---:|\n| `ENUM1` |\n| `ENUM2` |\n\n'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'member'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'TestClass'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'test'
    var_11 = 'test.TestClass'
    var_12 = '# test\n\n*Full name:* `test.TestClass`\n\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__attr_returns_none_for_invalid_attribute_path. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'invalid.attr.path'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 7/11 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 5/9 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias.value'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'name'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []



# Parsed testcases at query #10
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
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = var_1.toc
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 5/12 statements.
# Partially parsed test_visit_Attribute_keeps_non_typing_prefix. Retrieved 5/12 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 5/9 statements.
# Partially parsed test_class_api_with_enum_members. Retrieved 9/22 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 9/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 7/18 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 7/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = []
    var_6 = 'Bases'
    var_7 = bool('Bases' in var_0.doc[var_2])
    assert var_7 is True
    var_8 = '| BaseClass |'
    var_9 = bool('| BaseClass |' in var_0.doc[var_2])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.EnumClass'
    var_3 = 'enum.Enum'
    var_4 = []
    var_5 = 'MEMBER1'
    var_6 = 1
    var_7 = []
    var_8 = None
    var_9 = 'MEMBER2'
    var_10 = 2
    var_11 = []
    var_12 = 'Enums'
    var_13 = bool('Enums' in var_0.doc[var_2])
    assert var_13 is True
    var_14 = '| MEMBER1 |'
    var_15 = bool('| MEMBER1 |' in var_0.doc[var_2])
    assert var_15 is True
    var_16 = '| MEMBER2 |'
    var_17 = bool('| MEMBER2 |' in var_0.doc[var_2])
    assert var_17 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 'another_attr'
    var_9 = 42
    var_10 = []
    var_11 = 'Members'
    var_12 = bool('Members' in var_0.doc[var_2])
    assert var_12 is True
    var_13 = '| public_attr |'
    var_14 = bool('| public_attr |' in var_0.doc[var_2])
    assert var_14 is True
    var_15 = '| int |'
    var_16 = bool('| int |' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = '| another_attr |'
    var_18 = bool('| another_attr |' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc[var_2])
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.Class'
    var_3 = []
    var_4 = '_private_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = None
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc[var_2])
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.function'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = '_local'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private.public'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

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



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    '''Module docstring.'''\n    def foo():\n        '''Function docstring.'''\n        pass\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.foo'
    var_9 = bool('test_module.foo' in var_0.docstring)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    import os\n    from sys import path\n    def bar():\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_0.alias)
    assert var_5 is True
    var_6 = 'path'
    var_7 = bool('path' in var_0.alias)
    assert var_7 is True
    var_8 = 'test_module.bar'
    var_9 = bool('test_module.bar' in var_0.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    CONST = 42\n    x: int = 10\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.CONST'
    var_5 = bool('test_module.CONST' in var_0.const)
    assert var_5 is True
    var_6 = 'test_module.x'
    var_7 = bool('test_module.x' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    class MyClass:\n        '''Class docstring.'''\n        def method(self):\n            pass\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method'
    var_7 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.MyClass'
    var_9 = bool('test_module.MyClass' in var_0.docstring)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    @decorator\n    def decorated_func():\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.decorated_func'
    var_5 = bool('test_module.decorated_func' in var_0.doc)
    assert var_5 is True
    var_6 = '@decorator'
    var_7 = bool('@decorator' in var_0.doc['test_module.decorated_func'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    try:\n        def inner_func():\n            pass\n    except:\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.inner_func'
    var_5 = bool('test_module.inner_func' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    class Outer:\n        class Inner:\n            pass\n    '
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
    var_1 = '\n    x: int = 5\n    y: str\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.y'
    var_7 = bool('test_module.y' not in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = "\n    __all__ = ['public_func']\n    def public_func():\n        pass\n    def _private_func():\n        pass\n    "
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.public_func'
    var_5 = bool('test_module.public_func' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module._private_func'
    var_7 = bool('test_module._private_func' not in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'def func(): pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = '<a id="test_module.func"></a>'
    var_6 = bool('<a id="test_module.func"></a>' in var_1.doc['test_module.func'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = 'def func(): pass'
    var_3 = 'test_module'
    var_4 = var_1.parse(var_3, var_2)
    var_5 = var_1.link
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_const_type_constant. Retrieved 3/11 statements.
# Partially parsed test_const_type_tuple. Retrieved 4/21 statements.
# Partially parsed test_const_type_list. Retrieved 4/21 statements.
# Partially parsed test_const_type_set. Retrieved 4/21 statements.
# Partially parsed test_const_type_dict. Retrieved 6/24 statements.
# Partially parsed test_const_type_call. Retrieved 16/40 statements.
# Partially parsed test_const_type_other. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'hello'
    var_4 = [var_3]
    var_5 = True
    var_6 = [var_5]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'hello'
    var_8 = [var_7]
    var_9 = []
    var_10 = [var_9]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'hello'
    var_8 = [var_7]
    var_9 = []
    var_10 = [var_9]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'hello'
    var_8 = [var_7]
    var_9 = []
    var_10 = [var_9]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = [var_0]
    var_9 = [var_4]
    var_10 = [var_2]
    var_11 = [var_6]
    var_12 = []
    var_13 = []
    var_14 = [var_12, var_13]

def test_case_0():
    var_0 = 'bool'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'int'
    var_4 = [var_3]
    var_5 = []
    var_6 = 'float'
    var_7 = [var_6]
    var_8 = []
    var_9 = 'complex'
    var_10 = [var_9]
    var_11 = []
    var_12 = 'str'
    var_13 = [var_12]
    var_14 = []
    var_15 = 'list'
    var_16 = [var_15]
    var_17 = []
    var_18 = 'dict'
    var_19 = [var_18]
    var_20 = []
    var_21 = 'unknown'
    var_22 = [var_21]
    var_23 = []

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'y'



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x = 1'
    var_2 = 'test'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test'
    var_5 = bool('test' in var_0.doc)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_enums_not_empty. Retrieved 11/41 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'Enum'
    var_3 = []
    var_4 = 'VALUE1'
    var_5 = 'int'
    var_6 = []
    var_7 = 1
    var_8 = []
    var_9 = 'VALUE2'
    var_10 = []
    var_11 = 2
    var_12 = []
    var_13 = []
    var_14 = 'test_module'
    var_15 = 'test_module.TestClass'
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc['test_module.TestClass'])
    assert var_17 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_globals_predicate_false. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'y'
    var_4 = []
    var_5 = 1
    var_6 = []
    var_7 = 'root'
    var_8 = var_0.alias
    var_9 = bool(var_0.alias == {})
    assert var_9 is True
    var_10 = var_0.root
    var_11 = bool(var_0.root == {})
    assert var_11 is True
    var_12 = var_0.const
    var_13 = bool(var_0.const == {})
    assert var_13 is True
    var_14 = set()
    var_15 = {var_7: var_14}
    var_16 = var_0.imp
    var_17 = bool(var_0.imp == var_15)
    assert var_17 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 6/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 5/11 statements.
# Partially parsed test_globals_with_assign_no_type_comment. Retrieved 4/10 statements.
# Partially parsed test_globals_with__all__. Retrieved 4/12 statements.
# Partially parsed test_globals_with_non_constant__all__. Retrieved 5/14 statements.
# Partially parsed test_globals_with_non_name_target. Retrieved 4/12 statements.
# Partially parsed test_globals_with_non_uppercase_name. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = []
    var_4 = 'int'
    var_5 = []
    var_6 = 42
    var_7 = []
    var_8 = 1
    var_9 = var_0.alias['test_module.test_var']
    assert var_9 == '42'
    var_10 = var_0.const['test_module.test_var']
    assert var_10 == 'int'
    var_11 = var_0.root['test_module.test_var']
    var_12 = bool(var_0.root['test_module.test_var'] == var_1)
    assert var_12 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = var_0.alias['test_module.test_var']
    assert var_6 == '42'
    var_7 = var_0.const['test_module.test_var']
    assert var_7 == 'int'
    var_8 = var_0.root['test_module.test_var']
    var_9 = bool(var_0.root['test_module.test_var'] == var_1)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = 42
    var_4 = []
    var_5 = var_0.alias['test_module.test_var']
    assert var_5 == '42'
    var_6 = var_0.const['test_module.test_var']
    assert var_6 == 'int'
    var_7 = var_0.root['test_module.test_var']
    var_8 = bool(var_0.root['test_module.test_var'] == var_1)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'public_func'
    var_4 = []
    var_5 = var_0.imp[var_1]
    var_6 = bool(var_0.imp[var_1] == {'test_module.public_func'})
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'public_func'
    var_4 = []
    var_5 = set()
    var_6 = var_0.imp[var_1]
    var_7 = bool(var_0.imp[var_1] == var_5)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = 42
    var_4 = []
    var_5 = 'test_module.test_var'
    var_6 = bool('test_module.test_var' not in var_0.alias)
    assert var_6 is True
    var_7 = 'test_module.test_var'
    var_8 = bool('test_module.test_var' not in var_0.const)
    assert var_8 is True
    var_9 = 'test_module.test_var'
    var_10 = bool('test_module.test_var' not in var_0.root)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_var'
    var_3 = 42
    var_4 = []
    var_5 = var_0.alias['test_module.test_var']
    assert var_5 == '42'
    var_6 = 'test_module.test_var'
    var_7 = bool('test_module.test_var' not in var_0.const)
    assert var_7 is True
    var_8 = 'test_module.test_var'
    var_9 = bool('test_module.test_var' not in var_0.root)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'class MyEnum(enum.Enum):\n    A = 1\n    B = 2'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'Enums'
    var_5 = bool('Enums' in var_0.doc['pkg.MyEnum'])
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_api_function. Retrieved 25/38 statements.
# Partially parsed test_api_with_decorators. Retrieved 12/23 statements.
# Partially parsed test_api_with_docstring. Retrieved 12/23 statements.
# Partially parsed test_api_with_prefix. Retrieved 17/29 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'test_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = None
    var_13 = 'test_func'
    var_14 = 'async_func'
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = []
    var_23 = 'async_func'
    var_24 = 'TestClass'
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = []
    var_30 = 'TestClass'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'decorated_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'decorator'
    var_12 = []
    var_13 = None
    var_14 = 'decorated_func'
    var_15 = 'Decorators'
    var_16 = 'decorator'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'doc_func'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'This is a docstring'
    var_11 = []
    var_12 = []
    var_13 = None
    var_14 = 'doc_func'
    var_15 = 'This is a docstring'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_module'
    var_3 = 'TestClass'
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = 'method'
    var_10 = []
    var_11 = 'self'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = 'TestClass.method'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_ann_with_has_self_and_cls_method. Retrieved 7/14 statements.
# Partially parsed test_func_ann_with_has_self_no_cls_method. Retrieved 8/15 statements.
# Partially parsed test_func_ann_with_star_arg. Retrieved 7/13 statements.
# Partially parsed test_func_ann_with_no_annotation. Retrieved 5/9 statements.
# Partially parsed test_func_ann_with_self_type_in_cls_method. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
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
    var_3 = [var_1, var_2]
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
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = '*'
    var_5 = [var_4, var_2]
    var_6 = 'y'
    var_7 = [var_6, var_2]
    var_8 = 'root'
    var_9 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'root'
    var_5 = False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = 'root'
    var_5 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 4/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 7/11 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 5/9 statements.
# Partially parsed test_visit_Name_with_TypeVar. Retrieved 7/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'Self'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = 'name'
    var_5 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    var_6 = 'name'
    var_7 = []



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_chained_nonexistent. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'simple_attr'

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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_globals_with_ann_assign_and_constant. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign_and_constant. Retrieved 4/9 statements.
# Partially parsed test_globals_with_assign_and_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign_and_list. Retrieved 5/13 statements.
# Partially parsed test_globals_with_assign_and_tuple. Retrieved 5/13 statements.
# Partially parsed test_globals_with_assign_and_non_constant. Retrieved 4/9 statements.
# Partially parsed test_globals_with_ann_assign_and_non_constant. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign_and_multiple_targets. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = []
    var_3 = 'int'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'test.module'
    var_8 = var_0.alias['test.module.MY_CONST']
    assert var_8 == '42'
    var_9 = var_0.const['test.module.MY_CONST']
    assert var_9 == 'int'
    var_10 = var_0.root['test.module.MY_CONST']
    assert var_10 == 'test.module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = 'test.module'
    var_6 = var_0.alias['test.module.MY_CONST']
    assert var_6 == '42'
    var_7 = var_0.const['test.module.MY_CONST']
    assert var_7 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'MY_CONST'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = 'int'
    var_6 = 'test.module'
    var_7 = var_0.alias['test.module.MY_CONST']
    assert var_7 == '42'
    var_8 = var_0.const['test.module.MY_CONST']
    assert var_8 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = 'test.module'
    var_8 = var_0.imp['test.module']
    var_9 = bool(var_0.imp['test.module'] == {'test.module.func1', 'test.module.func2'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = 'test.module'
    var_8 = var_0.imp['test.module']
    var_9 = bool(var_0.imp['test.module'] == {'test.module.func1', 'test.module.func2'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'my_var'
    var_2 = []
    var_3 = 'some_value'
    var_4 = []
    var_5 = 'test.module'
    var_6 = var_0.alias['test.module.my_var']
    assert var_6 == 'some_value'
    var_7 = 'test.module.my_var'
    var_8 = bool('test.module.my_var' not in var_0.const)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'my_var'
    var_2 = []
    var_3 = 'int'
    var_4 = []
    var_5 = 'some_value'
    var_6 = []
    var_7 = 'test.module'
    var_8 = var_0.alias['test.module.my_var']
    assert var_8 == 'some_value'
    var_9 = 'test.module.my_var'
    var_10 = bool('test.module.my_var' not in var_0.const)
    assert var_10 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'my_var1'
    var_2 = []
    var_3 = 'my_var2'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'test.module'
    var_8 = 'test.module.my_var1'
    var_9 = bool('test.module.my_var1' not in var_0.alias)
    assert var_9 is True
    var_10 = 'test.module.my_var2'
    var_11 = bool('test.module.my_var2' not in var_0.alias)
    assert var_11 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_const_type_constant. Retrieved 3/9 statements.
# Partially parsed test_const_type_tuple. Retrieved 3/18 statements.
# Partially parsed test_const_type_list. Retrieved 3/18 statements.
# Partially parsed test_const_type_set. Retrieved 3/18 statements.
# Partially parsed test_const_type_dict. Retrieved 4/24 statements.
# Partially parsed test_const_type_call. Retrieved 7/17 statements.
# Partially parsed test_const_type_any. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 3.14
    var_3 = [var_2]
    var_4 = 'hello'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'a'
    var_8 = [var_7]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'a'
    var_8 = [var_7]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = [var_2]
    var_6 = [var_0]
    var_7 = 'a'
    var_8 = [var_7]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = 2.0
    var_6 = [var_5]
    var_7 = [var_0]
    var_8 = [var_5]
    var_9 = [var_2]
    var_10 = 'b'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'bool'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'int'
    var_4 = [var_3]
    var_5 = []
    var_6 = 'x'
    var_7 = [var_6]
    var_8 = 'float'
    var_9 = []

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'y'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_line_26_evaluates_to_false. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = 'public_attr'
    var_5 = []
    var_6 = 42
    var_7 = []
    var_8 = None
    var_9 = 'public_attr'
    var_10 = bool('public_attr' not in var_0.doc[var_2])
    assert var_10 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_func_api_with_posonlyargs. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_vararg. Retrieved 12/15 statements.
# Partially parsed test_func_api_with_kwonlyargs. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_kwarg. Retrieved 12/15 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 13/21 statements.
# Partially parsed test_func_api_with_has_self. Retrieved 12/16 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'b'
    var_5 = [var_4, var_2]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | / | return |\n|:---:|:---:|:---:|:---:|\n| `Any` | `Any` |  | `Any` |\n\n'

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
    var_8 = [var_7, var_6]
    var_9 = None
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'a'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = 'b'
    var_8 = [var_7, var_5]
    var_9 = []
    var_10 = None
    var_11 = 'root'
    var_12 = 'name'
    var_13 = False
    var_14 = var_0.doc['name']
    assert var_14 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n'

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
    var_8 = [var_6, var_7]
    var_9 = None
    var_10 = 'root'
    var_11 = 'name'
    var_12 = False
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'b'
    var_6 = [var_5, var_3]
    var_7 = 1
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = None
    var_14 = 'root'
    var_15 = 'name'
    var_16 = False
    var_17 = var_0.doc['name']
    assert var_17 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n| `1` | `2` |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'name'
    var_11 = True
    var_12 = False
    var_13 = var_0.doc['name']
    assert var_13 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:---:|:---:|\n| `Self` | `Any` |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = None
    var_9 = 'root'
    var_10 = 'name'
    var_11 = True
    var_12 = var_0.doc['name']
    assert var_12 == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| cls | return |\n|:---:|:---:|\n| `type[Self]` | `Any` |\n\n'



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def foo(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'root.foo'
    var_8 = bool('root.foo' in var_0.doc)
    assert var_8 is True
    var_9 = '## foo()'
    var_10 = bool('## foo()' in var_0.doc['root.foo'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'async def bar(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'root.bar'
    var_8 = bool('root.bar' in var_0.doc)
    assert var_8 is True
    var_9 = '## async bar()'
    var_10 = bool('## async bar()' in var_0.doc['root.bar'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Baz: pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'root.Baz'
    var_8 = bool('root.Baz' in var_0.doc)
    assert var_8 is True
    var_9 = '## class Baz'
    var_10 = bool('## class Baz' in var_0.doc['root.Baz'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Outer:\n    def inner(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.body[var_3]
    var_6 = 'root'
    var_7 = 'Outer'
    var_8 = var_0.api(var_6, var_5, prefix=var_7)
    var_9 = 'root.Outer.inner'
    var_10 = bool('root.Outer.inner' in var_0.doc)
    assert var_10 is True
    var_11 = '### inner()'
    var_12 = bool('### inner()' in var_0.doc['root.Outer.inner'])
    assert var_12 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '@decorator\ndef decorated(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'Decorators'
    var_8 = bool('Decorators' in var_0.doc['root.decorated'])
    assert var_8 is True
    var_9 = '| @decorator |'
    var_10 = bool('| @decorator |' in var_0.doc['root.decorated'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'def documented():\n    """This is a docstring."""\n    pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'root.documented'
    var_8 = bool('root.documented' in var_0.docstring)
    assert var_8 is True
    var_9 = 'This is a docstring.'
    var_10 = bool('This is a docstring.' in var_0.docstring['root.documented'])
    assert var_10 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'class Container:\n    def method(): pass'
    var_2 = module_1.parse(var_1)
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'root'
    var_6 = var_0.api(var_5, var_4)
    var_7 = 'root.Container'
    var_8 = bool('root.Container' in var_0.doc)
    assert var_8 is True
    var_9 = 'root.Container.method'
    var_10 = bool('root.Container.method' in var_0.doc)
    assert var_10 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_api_function_without_prefix. Retrieved 11/15 statements.
# Partially parsed test_api_function_with_prefix. Retrieved 12/16 statements.
# Partially parsed test_api_async_function. Retrieved 11/15 statements.
# Partially parsed test_api_class. Retrieved 8/11 statements.
# Partially parsed test_api_with_decorators. Retrieved 11/18 statements.
# Partially parsed test_api_with_link_enabled. Retrieved 11/15 statements.
# Partially parsed test_api_class_with_bases. Retrieved 8/14 statements.
# Partially parsed test_api_class_with_enums. Retrieved 10/27 statements.
# Partially parsed test_api_class_with_members. Retrieved 12/28 statements.
# Partially parsed test_api_class_with_deleted_member. Retrieved 9/23 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test_func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'Class'
    var_13 = 'root.Class.test_func'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_async'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test_async'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'root.TestClass'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'decorator'
    var_3 = []
    var_4 = 'test_func'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = None
    var_12 = 'root'
    var_13 = 'root.test_func'
    var_14 = 'Decorators'
    var_15 = '| decorator |'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'root'
    var_12 = 'root.test_func'
    var_13 = '<a id="root-test_func"></a>'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'BaseClass'
    var_3 = []
    var_4 = 'TestClass'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'root'
    var_9 = 'root.TestClass'
    var_10 = 'Bases'
    var_11 = '| BaseClass |'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'enum.Enum'
    var_3 = []
    var_4 = 'TestEnum'
    var_5 = []
    var_6 = 'VALUE1'
    var_7 = []
    var_8 = []
    var_9 = 'VALUE2'
    var_10 = 2
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = 'root'
    var_15 = 'root.TestEnum'
    var_16 = 'Enums'
    var_17 = '| VALUE1 |'
    var_18 = '| VALUE2 |'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = 'member1'
    var_6 = 'int'
    var_7 = []
    var_8 = []
    var_9 = 'member2'
    var_10 = 'str'
    var_11 = []
    var_12 = 'test'
    var_13 = []
    var_14 = []
    var_15 = 'root'
    var_16 = 'root.TestClass'
    var_17 = 'Members'
    var_18 = '| member1 | int |'
    var_19 = '| member2 | str |'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = 'member1'
    var_6 = 'int'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'root.TestClass'
    var_12 = 'Members'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_globals_with_annassign. Retrieved 5/10 statements.
# Partially parsed test_globals_with_assign. Retrieved 4/9 statements.
# Partially parsed test_globals_with_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 5/13 statements.
# Partially parsed test_globals_ignores_non_constant. Retrieved 4/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = []
    var_3 = 'int'
    var_4 = []
    var_5 = 42
    var_6 = []
    var_7 = 'module'
    var_8 = var_0.alias['module.VAR']
    assert var_8 == '42'
    var_9 = var_0.const['module.VAR']
    assert var_9 == 'int'
    var_10 = var_0.root['module.VAR']
    assert var_10 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = 'module'
    var_6 = var_0.alias['module.VAR']
    assert var_6 == '42'
    var_7 = var_0.const['module.VAR']
    assert var_7 == 'int'
    var_8 = var_0.root['module.VAR']
    assert var_8 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = []
    var_3 = 42
    var_4 = []
    var_5 = 'float'
    var_6 = 'module'
    var_7 = var_0.alias['module.VAR']
    assert var_7 == '42'
    var_8 = var_0.const['module.VAR']
    assert var_8 == 'float'
    var_9 = var_0.root['module.VAR']
    assert var_9 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'func1'
    var_4 = []
    var_5 = 'func2'
    var_6 = []
    var_7 = 'module'
    var_8 = var_0.imp['module']
    var_9 = bool(var_0.imp['module'] == {'module.func1', 'module.func2'})
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'VAR'
    var_2 = []
    var_3 = 'other_var'
    var_4 = []
    var_5 = 'module'
    var_6 = 'module.VAR'
    var_7 = bool('module.VAR' not in var_0.alias)
    assert var_7 is True
    var_8 = 'module.VAR'
    var_9 = bool('module.VAR' not in var_0.const)
    assert var_9 is True
    var_10 = 'module.VAR'
    var_11 = bool('module.VAR' not in var_0.root)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 4/8 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 5/9 statements.
# Partially parsed test_is_public_with_all_listed_name. Retrieved 6/10 statements.
# Partially parsed test_is_public_with_not_listed_in_all. Retrieved 7/11 statements.
# Partially parsed test_is_public_with_const_in_all. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'public_name'
    var_2 = ''
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '_private_name'
    var_2 = ''
    var_3 = var_0.is_public(var_1)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__magic__'
    var_2 = ''
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent.public_child'
    var_2 = ''
    var_3 = 'parent'
    var_4 = var_0.is_public(var_1)
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent._private_child'
    var_2 = ''
    var_3 = 'parent'
    var_4 = var_0.is_public(var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent'
    var_2 = 'parent.child'
    var_3 = {var_2}
    var_4 = ''
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent'
    var_2 = 'parent.other'
    var_3 = {var_2}
    var_4 = 'parent.child'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'parent'
    var_2 = 'parent.CONST'
    var_3 = {var_2}
    var_4 = 'int'
    var_5 = var_0.is_public(var_2)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [' '])
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' '])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['`x`'])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == ['`x`', '`y`', '`z`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x&y'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['<code>x&#38;y</code>'])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x|y'
    var_1 = [var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['`<code>x&#124;y</code>`'])
    assert var_4 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = None
    var_2 = 'y&z'
    var_3 = 'a|b'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._defaults(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == ['`x`', ' ', '<code>y&#38;z</code>', '`<code>a&#124;b</code>`'])
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_api_function. Retrieved 5/6 statements.
# Partially parsed test_api_async_function. Retrieved 5/6 statements.
# Partially parsed test_api_class. Retrieved 5/6 statements.
# Partially parsed test_api_with_prefix. Retrieved 5/6 statements.
# Partially parsed test_api_with_decorator. Retrieved 5/6 statements.
# Partially parsed test_api_with_docstring. Retrieved 5/6 statements.
# Partially parsed test_api_nested_class. Retrieved 5/6 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'def func(): pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'func'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'async def func(): pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'func'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'class MyClass: pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'MyClass'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'class MyClass:\n    def method(self): pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'MyClass.method'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = '@decorator\ndef func(): pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'func'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'def func():\n    """This is a docstring."""\n    pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'func'
    var_6 = 'func'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test'
    var_3 = 'class Outer:\n    class Inner: pass'
    var_4 = module_0.parse(var_2, var_3)
    var_5 = 'Outer.Inner'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test__e_type_empty_input.
# Partially parsed test__e_type_none_element. Retrieved 2/3 statements.
# Partially parsed test__e_type_non_constant_element. Retrieved 3/4 statements.
# Partially parsed test__e_type_single_type. Retrieved 2/6 statements.
# Partially parsed test__e_type_mixed_types. Retrieved 2/6 statements.
# Partially parsed test__e_type_multiple_sequences. Retrieved 2/7 statements.
# Partially parsed test__e_type_mixed_sequences. Retrieved 2/7 statements.
# Partially parsed test__e_type_all_any. Retrieved 4/11 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
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
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = 2
    var_5 = [var_4]
    var_6 = 3.0
    var_7 = [var_6]



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = ''
    var_3 = var_0.parse(var_1, var_2)
    var_4 = var_0.doc['test_module']
    assert var_4 == '### Module `{}`'
    var_5 = var_0.level['test_module']
    assert var_5 == 0
    var_6 = set()
    var_7 = var_0.imp['test_module']
    var_8 = bool(var_0.imp['test_module'] == var_6)
    assert var_8 is True
    var_9 = var_0.root['test_module']
    assert var_9 == 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    """\n    This is a test module.\n    """\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc['test_module']
    assert var_4 == '### Module `{}`'
    var_5 = var_0.docstring['test_module']
    assert var_5 == '```python\n    """\n    This is a test module.\n    """\n```'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    import os\n    from sys import path\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.alias['test_module.os']
    assert var_4 == 'os'
    var_5 = var_0.alias['test_module.path']
    assert var_5 == 'sys.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    CONSTANT = 42\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.alias['test_module.CONSTANT']
    assert var_4 == '42'
    var_5 = var_0.root['test_module.CONSTANT']
    assert var_5 == 'test_module'
    var_6 = var_0.const['test_module.CONSTANT']
    assert var_6 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    def test_function():\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc['test_module.test_function']
    assert var_4 == '#### test_function()\n\n*Full name:* `{}`'
    var_5 = var_0.level['test_module.test_function']
    assert var_5 == 0
    var_6 = var_0.root['test_module.test_function']
    assert var_6 == 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    class TestClass:\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = var_0.doc['test_module.TestClass']
    assert var_4 == '#### class TestClass\n\n*Full name:* `{}`'
    var_5 = var_0.level['test_module.TestClass']
    assert var_5 == 0
    var_6 = var_0.root['test_module.TestClass']
    assert var_6 == 'test_module'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_walk_body_single_statement.
# Failed to parse test_walk_body_multiple_statements.
# Partially parsed test_walk_body_if_statement. Retrieved 1/13 statements.
# Partially parsed test_walk_body_nested_if_statements. Retrieved 3/18 statements.
# Failed to parse test_walk_body_try_statement.
# Partially parsed test_walk_body_mixed_statements. Retrieved 4/20 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'y'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'x'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_empty_elements.




# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_class_api. Retrieved 19/42 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = bool(var_2 in var_0.doc)
    assert var_6 is True
    var_7 = var_0.doc[var_2]
    assert var_7 == '### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n'
    var_8 = 'BaseClass'
    var_9 = []
    var_10 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_11 = 'Bases'
    var_12 = bool('Bases' in var_0.doc[var_2])
    assert var_12 is True
    var_13 = '| BaseClass |'
    var_14 = bool('| BaseClass |' in var_0.doc[var_2])
    assert var_14 is True
    var_15 = 'attr1'
    var_16 = 'int'
    var_17 = []
    var_18 = None
    var_19 = 1
    var_20 = '_private_attr'
    var_21 = 'str'
    var_22 = []
    var_23 = 'attr2'
    var_24 = 42
    var_25 = []
    var_26 = 'float'
    var_27 = []
    var_28 = var_0.class_api(var_1, var_2, var_27, var_4)
    var_29 = 'Members'
    var_30 = bool('Members' in var_0.doc[var_2])
    assert var_30 is True
    var_31 = '| attr2 | float |'
    var_32 = bool('| attr2 | float |' in var_0.doc[var_2])
    assert var_32 is True
    var_33 = '| attr1 |'
    var_34 = bool('| attr1 |' not in var_0.doc[var_2])
    assert var_34 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 5/11 statements.
# Partially parsed test_visit_Attribute_non_typing_prefix. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = 'List'
    var_6 = []

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = []
    var_5 = 'attr'
    var_6 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test_attr_missing_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_missing_nested_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'

def test_case_0():
    var_0 = 'nested.value'

def test_case_0():
    var_0 = 'missing'

def test_case_0():
    var_0 = 'nested.missing'

def test_case_0():
    var_0 = ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'value'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_api_with_empty_prefix. Retrieved 11/14 statements.


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
    var_10 = 'root'
    var_11 = ''
    var_12 = bool(not var_0.api.__code__.co_varnames[1] == 'prefix')
    assert var_12 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_isinstance_node_Import_ImportFrom. Retrieved 6/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os\nfrom sys import path'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = True
    var_5 = module_1.parse(var_1, type_comments=var_4)
    var_6 = bool(var_3)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__attr_returns_none_when_intermediate_attribute_is_none. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'a.b'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 11/18 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/16 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 11/14 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 12/20 statements.
# Partially parsed test_func_api_with_self. Retrieved 12/17 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 11/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = 'b'
    var_5 = [var_4, var_2]
    var_6 = 'c'
    var_7 = [var_6, var_2]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'root'
    var_12 = 'func'
    var_13 = False
    var_14 = var_0.doc['root.func']
    assert var_14 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| a | b | / | c | return |\n|:---:|:---:|:---:|:---:|:---:|\n| ANY | ANY |  | ANY | ANY |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'a'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = 'b'
    var_8 = [var_7, var_5]
    var_9 = []
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False
    var_13 = var_0.doc['root.func']
    assert var_13 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | ANY | ANY | ANY |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'args'
    var_9 = [var_8, var_3]
    var_10 = 'root'
    var_11 = 'func'
    var_12 = False
    var_13 = var_0.doc['root.func']
    assert var_13 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| a | *args | return |\n|:---:|:---:|:---:|\n| ANY |  | ANY |\n\n'

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
    var_8 = [var_6, var_7]
    var_9 = 'root'
    var_10 = 'func'
    var_11 = False
    var_12 = var_0.doc['root.func']
    assert var_12 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| **kwargs | return |\n|:---:|:---:|\n|  | ANY |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'b'
    var_6 = [var_5, var_3]
    var_7 = 1
    var_8 = []
    var_9 = 2
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'func'
    var_15 = False
    var_16 = var_0.doc['root.func']
    assert var_16 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| a | b | return |\n|:---:|:---:|:---:|\n| ANY | ANY | ANY |\n| 1 | 2 |  |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = [var_5, var_3]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func'
    var_12 = True
    var_13 = False
    var_14 = var_0.doc['root.func']
    assert var_14 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| Self | a | return |\n|:---:|:---:|:---:|\n| ANY | ANY | ANY |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = [var_5, var_3]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 'root'
    var_11 = 'func'
    var_12 = True
    var_13 = var_0.doc['root.func']
    assert var_13 == '### func()\n\n*Full name:* `root.func`\n<a id="root-func"></a>\n\n| type[Self] | a | return |\n|:---:|:---:|:---:|\n| ANY | ANY | ANY |\n\n'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_annassign_name_isinstance. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_public_predicate_false. Retrieved 6/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'child'
    var_3 = {var_2}
    var_4 = 'root.child'
    var_5 = var_0.is_public(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_attr_single_level. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_levels. Retrieved 1/10 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_intermediate_none. Retrieved 1/8 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'

def test_case_0():
    var_0 = 'inner.data'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'inner.data'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_class_api_with_bases. Retrieved 10/22 statements.
# Partially parsed test_class_api_with_enum. Retrieved 10/27 statements.
# Partially parsed test_class_api_with_members. Retrieved 12/31 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'pkg'
    var_3 = 'pkg.Base'
    var_4 = 'base.Base'
    var_5 = '# Module `pkg`\n\n'
    var_6 = 'Child'
    var_7 = 'Base'
    var_8 = []
    var_9 = []
    var_10 = 'pkg.Child'
    var_11 = 'pkg.Child'
    var_12 = 'Bases'
    var_13 = 'base.Base'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'pkg'
    var_3 = '# Module `pkg`\n\n'
    var_4 = 'Color'
    var_5 = 'Enum'
    var_6 = []
    var_7 = 'RED'
    var_8 = 'int'
    var_9 = []
    var_10 = None
    var_11 = 'pkg.Color'
    var_12 = 'Enums'
    var_13 = 'RED'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'pkg'
    var_3 = '# Module `pkg`\n\n'
    var_4 = 'MyClass'
    var_5 = []
    var_6 = 'attr1'
    var_7 = 'int'
    var_8 = []
    var_9 = None
    var_10 = 'attr2'
    var_11 = 42
    var_12 = []
    var_13 = 'pkg.MyClass'
    var_14 = 'Members'
    var_15 = 'attr1'
    var_16 = 'attr2'



# Parsed testcases at query #24
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = 'simple'
    var_21 = {var_17: var_5, var_18: var_10, var_19: var_15, var_20: var_11}
    var_22 = module_1.AnnAssign(*var_16, **var_21)
    var_23 = 'root'
    var_24 = var_0.globals(var_23, var_22)
    var_25 = var_0.alias['root.x']
    assert var_25 == '1'
    var_26 = var_0.const['root.x']
    assert var_26 == 'int'
    var_27 = var_0.root['root.x']
    assert var_27 == 'root'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'y'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'hello'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'root'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['root.y']
    assert var_19 == "'hello'"
    var_20 = var_0.const['root.y']
    assert var_20 == 'str'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'func1'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'func2'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = {var_19: var_17}
    var_21 = module_1.List(*var_18, **var_20)
    var_22 = []
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_21}
    var_26 = module_1.Assign(*var_22, **var_25)
    var_27 = 'root'
    var_28 = var_0.globals(var_27, var_26)
    var_29 = var_0.imp['root']
    var_30 = bool(var_0.imp['root'] == {'root.func1', 'root.func2'})
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'y'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = [var_5, var_10]
    var_12 = 1
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_11, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'root.x'
    var_25 = bool('root.x' not in var_0.alias)
    assert var_25 is True
    var_26 = 'root.y'
    var_27 = bool('root.y' not in var_0.alias)
    assert var_27 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'some_list'
    var_8 = []
    var_9 = 'id'
    var_10 = {var_9: var_7}
    var_11 = module_1.Name(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'root'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = set()
    var_20 = var_0.imp['root']
    var_21 = bool(var_0.imp['root'] == var_19)
    assert var_21 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_class_api. Retrieved 19/35 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'attr1'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = None
    var_24 = 'attr2'
    var_25 = 'str'
    var_26 = []
    var_27 = {}
    var_28 = module_1.Load(*var_26, **var_27)
    var_29 = []
    var_30 = 'id'
    var_31 = 'ctx'
    var_32 = {var_30: var_25, var_31: var_28}
    var_33 = module_1.Name(*var_29, **var_32)
    var_34 = 'attr3'
    var_35 = 42
    var_36 = []
    var_37 = 'value'
    var_38 = {var_37: var_35}
    var_39 = module_1.Constant(*var_36, **var_38)
    var_40 = var_0.doc[var_2]
    assert var_40 == '# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id="test-module-testclass"></a>\n\n| Bases |\n|:-----:|\n|<code>BaseClass</code>|\n\n| Members | Type |\n|:--------:|:-----:|\n|<code>attr1</code>|<code>int</code>|\n|<code>attr3</code>|<code>int</code>|\n\n'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_visit_Constant_with_valid_name. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 42
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Constant(*var_4, **var_5)
    var_7 = var_2.visit_Constant(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Constant(*var_6, **var_7)
    var_9 = var_4.visit_Constant(var_8)
    var_10 = var_9.id
    assert var_10 == 'alias'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_node_type_comment_is_not_none. Retrieved 11/12 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    var_19 = 'root'
    var_20 = var_0.globals(var_19, var_18)
    assert var_20 is None
    var_21 = 'root.x'
    var_22 = bool('root.x' in var_0.alias)
    assert var_22 is True
    var_23 = var_0.alias['root.x']
    assert var_23 == '42'
    var_24 = 'root.x'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_docstring. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = ''
    var_4 = None
    var_5 = var_0.docstring['pkg.module.func']
    assert var_5 == '```python\nThis is a function.\n```'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_api_function. Retrieved 13/17 statements.
# Partially parsed test_api_async_function. Retrieved 13/17 statements.
# Partially parsed test_api_class. Retrieved 11/12 statements.
# Partially parsed test_api_with_prefix. Retrieved 14/18 statements.
# Partially parsed test_api_with_decorators. Retrieved 13/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'test_module.test_func'
    var_12 = bool('test_module.test_func' in var_0.doc)
    assert var_12 is True
    var_13 = 'test_module.test_func'
    var_14 = var_0.doc[var_13]
    var_15 = '### test_func()'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'test_module.test_async_func'
    var_12 = bool('test_module.test_async_func' in var_0.doc)
    assert var_12 is True
    var_13 = 'test_module.test_async_func'
    var_14 = var_0.doc[var_13]
    var_15 = '### async test_async_func()'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'name'
    var_8 = 'bases'
    var_9 = 'body'
    var_10 = 'decorator_list'
    var_11 = {var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_1.ClassDef(*var_6, **var_11)
    var_13 = var_0.api(var_1, var_12)
    var_14 = 'test_module.TestClass'
    var_15 = bool('test_module.TestClass' in var_0.doc)
    assert var_15 is True
    var_16 = 'test_module.TestClass'
    var_17 = var_0.doc[var_16]
    var_18 = '### class TestClass'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'method'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'TestClass'
    var_12 = 'test_module.TestClass.method'
    var_13 = bool('test_module.TestClass.method' in var_0.doc)
    assert var_13 is True
    var_14 = 'test_module.TestClass.method'
    var_15 = var_0.doc[var_14]
    var_16 = '#### method()'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'decorator'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 'decorated_func'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = [var_10]
    var_20 = 'test_module.decorated_func'
    var_21 = bool('test_module.decorated_func' in var_0.doc)
    assert var_21 is True
    var_22 = 'Decorators'
    var_23 = bool('Decorators' in var_0.doc['test_module.decorated_func'])
    assert var_23 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/7 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_chain_break. Retrieved 1/6 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.
# Partially parsed test_attr_single_dot. Retrieved 1/7 statements.
# Partially parsed test_attr_trailing_dot. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'simple'

def test_case_0():
    var_0 = 'nested.attr'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'level1.level2.attr'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '.a.b'

def test_case_0():
    var_0 = 'a.'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_class_api_annassign_with_name_target. Retrieved 11/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'test_attr'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = 42
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = 1
    var_20 = var_0.doc[var_2]
    var_21 = bool(var_0.doc[var_2] == '#' * (var_0.b_level + 2) + ' class TestClass\n\n*Full name:* `test_module.TestClass`\n\n')
    assert var_21 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a regular line.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a regular line.'

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('hello')\n```"

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Line 1\nLine 2\nLine 3'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Line 1\nLine 2\nLine 3'

import apimd.parser as module_0

def test_case_0():
    var_0 = "Line 1\n>>> print('hello')\nLine 2"
    var_1 = "Line 1\n```python\n>>> print('hello')\n```\nLine 2"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "Line 1\n>>> print('hello')\nLine 2\n>>> x = 1\nLine 3"
    var_1 = "Line 1\n```python\n>>> print('hello')\n```\nLine 2\n```python\n>>> x = 1\n```\nLine 3"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = "Line 1\n>>> print('hello')"
    var_1 = "Line 1\n```python\n>>> print('hello')\n```"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')\nLine 1"
    var_1 = "```python\n>>> print('hello')\n```\nLine 1"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')\n>>> x = 1"
    var_1 = "```python\n>>> print('hello')\n>>> x = 1\n```"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')\nNormal line\n>>> x = 1\nAnother normal line"
    var_1 = "```python\n>>> print('hello')\n```\nNormal line\n```python\n>>> x = 1\n```\nAnother normal line"
    var_2 = module_0.doctest(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_listed_in_all. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_parent_listed_in_all. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_const. Retrieved 6/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = 'root.public_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root._private_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.__magic__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'nested'
    var_3 = {var_2}
    var_4 = 'root.nested.public_name'
    var_5 = ''
    var_6 = 'root.nested'
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.nested._private_name'
    var_4 = ''
    var_5 = 'root.nested'
    var_6 = var_0.is_public(var_3)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'listed_name'
    var_3 = {var_2}
    var_4 = 'root.listed_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'root.parent.child'
    var_5 = ''
    var_6 = 'root.parent'
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.CONST'
    var_4 = 'int'
    var_5 = var_0.is_public(var_3)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_visit_Attribute_removes_typing_prefix. Retrieved 11/13 statements.
# Partially parsed test_visit_Attribute_keeps_non_typing_attribute. Retrieved 11/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'typing'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'List'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = 'attr'
    var_19 = 'ctx'
    var_20 = {var_17: var_11, var_18: var_12, var_19: var_15}
    var_21 = module_1.Attribute(*var_16, **var_20)
    var_22 = var_2.visit_Attribute(var_21)
    var_23 = var_22.id
    assert var_23 == 'List'
    var_24 = var_22.ctx

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'other'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'List'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = 'attr'
    var_19 = 'ctx'
    var_20 = {var_17: var_11, var_18: var_12, var_19: var_15}
    var_21 = module_1.Attribute(*var_16, **var_20)
    var_22 = var_2.visit_Attribute(var_21)
    var_23 = var_22.value.id
    assert var_23 == 'other'
    var_24 = var_22.attr
    assert var_24 == 'List'
    var_25 = var_22.ctx



# Parsed testcases at query #12
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    """Module docstring."""\n    x = 1\n    def foo():\n        """Function docstring."""\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.foo'
    var_7 = bool('test_module.foo' in var_0.doc)
    assert var_7 is True
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    import os\n    from sys import path\n    x = 1\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.path'
    var_7 = bool('test_module.path' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    x: int = 1\n    y: str = "hello"\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.x'
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True
    var_6 = 'test_module.y'
    var_7 = bool('test_module.y' in var_0.alias)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    class MyClass:\n        """Class docstring."""\n        def method(self):\n            pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.MyClass'
    var_5 = bool('test_module.MyClass' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module.MyClass.method'
    var_7 = bool('test_module.MyClass.method' in var_0.doc)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    @decorator\n    def foo():\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = 'Decorator'
    var_7 = bool('Decorator' in var_0.doc['test_module.foo'])
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    CONSTANT = 42\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.CONSTANT'
    var_5 = bool('test_module.CONSTANT' in var_0.const)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '\n    __all__ = ["foo", "bar"]\n    def foo():\n        pass\n    def bar():\n        pass\n    def baz():\n        pass\n    '
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    var_4 = 'test_module.foo'
    var_5 = bool('test_module.foo' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'test_module.bar'
    var_7 = bool('test_module.bar' in var_0.imp['test_module'])
    assert var_7 is True
    var_8 = 'test_module.baz'
    var_9 = bool('test_module.baz' not in var_0.imp['test_module'])
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_public_returns_false_when_all_l_is_empty. Retrieved 3/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_name'
    var_2 = var_0.is_public(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_class_api_with_enum_members. Retrieved 15/23 statements.
# Partially parsed test_class_api_with_public_members. Retrieved 13/22 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/19 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = var_0.class_api(var_1, var_2, var_12, var_13)
    var_15 = 'Bases'
    var_16 = bool('Bases' in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'BaseClass'
    var_18 = bool('BaseClass' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'Enum'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'MEMBER1'
    var_14 = 1
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = []
    var_20 = 'value'
    var_21 = {var_20: var_14}
    var_22 = module_1.Constant(*var_19, **var_21)
    var_23 = 'MEMBER2'
    var_24 = 2
    var_25 = []
    var_26 = 'value'
    var_27 = {var_26: var_24}
    var_28 = module_1.Constant(*var_25, **var_27)
    var_29 = []
    var_30 = 'value'
    var_31 = {var_30: var_24}
    var_32 = module_1.Constant(*var_29, **var_31)
    var_33 = 'Enums'
    var_34 = bool('Enums' in var_0.doc[var_2])
    assert var_34 is True
    var_35 = 'MEMBER1'
    var_36 = bool('MEMBER1' in var_0.doc[var_2])
    assert var_36 is True
    var_37 = 'MEMBER2'
    var_38 = bool('MEMBER2' in var_0.doc[var_2])
    assert var_38 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_member'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = 42
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = 'another_member'
    var_20 = 'hello'
    var_21 = []
    var_22 = 'value'
    var_23 = {var_22: var_20}
    var_24 = module_1.Constant(*var_21, **var_23)
    var_25 = 'Members'
    var_26 = bool('Members' in var_0.doc[var_2])
    assert var_26 is True
    var_27 = 'public_member'
    var_28 = bool('public_member' in var_0.doc[var_2])
    assert var_28 is True
    var_29 = 'another_member'
    var_30 = bool('another_member' in var_0.doc[var_2])
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'public_member'
    var_5 = 'int'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = 42
    var_15 = []
    var_16 = 'value'
    var_17 = {var_16: var_14}
    var_18 = module_1.Constant(*var_15, **var_17)
    var_19 = 'public_member'
    var_20 = bool('public_member' not in var_0.doc[var_2])
    assert var_20 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__attr_returns_none_when_attribute_not_found. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_imports_with_Import_node. Retrieved 4/8 statements.
# Partially parsed test_imports_with_Import_node_and_asname. Retrieved 4/8 statements.
# Partially parsed test_imports_with_ImportFrom_node_and_level. Retrieved 6/10 statements.
# Partially parsed test_imports_with_ImportFrom_node_and_asname. Retrieved 6/10 statements.


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
    var_1 = 'test_module.sub_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['test_module.sub_module.path']
    assert var_6 == 'test_module.os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'os_path'
    var_5 = 0
    var_6 = var_0.alias['test_module.os_path']
    assert var_6 == 'os.path'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_compile_empty. Retrieved 2/4 statements.
# Partially parsed test_compile_with_toc. Retrieved 1/6 statements.
# Partially parsed test_compile_with_magic_name. Retrieved 2/7 statements.
# Partially parsed test_compile_with_docstring. Retrieved 2/8 statements.
# Partially parsed test_compile_with_const. Retrieved 3/10 statements.
# Partially parsed test_compile_with_non_public. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'CONST'

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #18
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_0, var_1, var_0, var_2]
    var_4 = module_0._defaults(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [' ', '`x`', ' ', '`y`'])
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [' ', ' '])
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a&b'
    var_1 = 'c|d'
    var_2 = [var_0, var_1]
    var_3 = module_0._defaults(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['<code>a&b</code>', '`c&#124;d`'])
    assert var_5 is True

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
    var_0 = []
    var_1 = module_0._defaults(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_class_api_with_enum. Retrieved 15/23 statements.
# Partially parsed test_class_api_with_members. Retrieved 13/21 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 9/18 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 9/14 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'BaseClass'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = 'root'
    var_13 = 'root.ChildClass'
    var_14 = var_0.class_api(var_12, var_13, var_10, var_11)
    var_15 = '| Bases |\n|:---:|\n| `BaseClass` |\n\n'
    var_16 = bool('| Bases |\n|:---:|\n| `BaseClass` |\n\n' in var_0.doc['root.ChildClass'])
    assert var_16 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = 'root'
    var_4 = 'root.ChildClass'
    var_5 = var_0.class_api(var_3, var_4, var_1, var_2)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc['root.ChildClass'])
    assert var_7 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'Enum'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_1, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = 'MEMBER1'
    var_12 = 1
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_12}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = 'MEMBER2'
    var_22 = 2
    var_23 = []
    var_24 = 'value'
    var_25 = {var_24: var_22}
    var_26 = module_1.Constant(*var_23, **var_25)
    var_27 = []
    var_28 = 'value'
    var_29 = {var_28: var_22}
    var_30 = module_1.Constant(*var_27, **var_29)
    var_31 = 'root'
    var_32 = 'root.MyEnum'
    var_33 = '| Enums |\n|:---:|\n| `MEMBER1` |\n| `MEMBER2` |\n\n'
    var_34 = bool('| Enums |\n|:---:|\n| `MEMBER1` |\n| `MEMBER2` |\n\n' in var_0.doc['root.MyEnum'])
    assert var_34 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = None
    var_13 = 'attr2'
    var_14 = 'str'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = 'root'
    var_24 = 'root.MyClass'
    var_25 = '| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n'
    var_26 = bool('| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n' in var_0.doc['root.MyClass'])
    assert var_26 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'attr1'
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = None
    var_13 = 'root'
    var_14 = 'root.MyClass'
    var_15 = 'Members'
    var_16 = bool('Members' not in var_0.doc['root.MyClass'])
    assert var_16 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = '_private'
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = None
    var_13 = 'root'
    var_14 = 'root.MyClass'
    var_15 = 'Members'
    var_16 = bool('Members' not in var_0.doc['root.MyClass'])
    assert var_16 is True



# Parsed testcases at query #20
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 0
    var_4 = 'enum.Enum'
    var_5 = module_1.parse(var_4)
    var_6 = var_5.body[var_3]
    var_7 = var_6.value
    var_8 = [var_7]
    var_9 = 'VALUE'
    var_10 = []
    var_11 = 'id'
    var_12 = {var_11: var_9}
    var_13 = module_1.Name(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = 1
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_15}
    var_19 = module_1.Constant(*var_16, **var_18)
    var_20 = []
    var_21 = 'targets'
    var_22 = 'value'
    var_23 = {var_21: var_14, var_22: var_19}
    var_24 = module_1.Assign(*var_20, **var_23)
    var_25 = [var_24]
    var_26 = var_0.class_api(var_1, var_2, var_8, var_25)
    var_27 = 'Enums'
    var_28 = bool('Enums' in var_0.doc[var_2])
    assert var_28 is True



# Parsed testcases at query #21
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.expr(*var_1, **var_2)
    var_4 = 'y'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.expr(*var_5, **var_6)
    var_8 = 'z'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.expr(*var_9, **var_10)
    var_12 = [var_3, var_7, var_11]
    var_13 = module_1._defaults(var_12)
    var_14 = list(var_13)
    var_15 = [var_0]
    var_16 = {}
    var_17 = module_0.expr(*var_15, **var_16)
    var_18 = module_0.unparse(var_17)
    var_19 = module_1.code(var_18)
    var_20 = [var_4]
    var_21 = {}
    var_22 = module_0.expr(*var_20, **var_21)
    var_23 = module_0.unparse(var_22)
    var_24 = module_1.code(var_23)
    var_25 = [var_8]
    var_26 = {}
    var_27 = module_0.expr(*var_25, **var_26)
    var_28 = module_0.unparse(var_27)
    var_29 = module_1.code(var_28)
    var_30 = [var_19, var_24, var_29]
    var_31 = bool(var_14 == var_30)
    assert var_31 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_walk_body_try_with_handler. Retrieved 25/31 statements.
# Partially parsed test_walk_body_mixed_statements. Retrieved 79/85 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = module_1.walk_body(var_16)
    var_18 = list(var_17)
    var_19 = bool(var_18 == [var_15])
    assert var_19 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'y'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = 2
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_10, var_18: var_15}
    var_20 = module_0.Assign(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = 'test'
    var_25 = 'body'
    var_26 = 'orelse'
    var_27 = {var_24: var_4, var_25: var_21, var_26: var_22}
    var_28 = module_0.If(*var_23, **var_27)
    var_29 = [var_28]
    var_30 = module_1.walk_body(var_29)
    var_31 = list(var_30)
    var_32 = []
    var_33 = 'id'
    var_34 = {var_33: var_5}
    var_35 = module_0.Name(*var_32, **var_34)
    var_36 = [var_35]
    var_37 = []
    var_38 = 'value'
    var_39 = {var_38: var_11}
    var_40 = module_0.Constant(*var_37, **var_39)
    var_41 = []
    var_42 = 'targets'
    var_43 = 'value'
    var_44 = {var_42: var_36, var_43: var_40}
    var_45 = module_0.Assign(*var_41, **var_44)
    var_46 = [var_45]
    var_47 = bool(var_31 == var_46)
    assert var_47 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'y'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = 2
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_0.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_10, var_18: var_15}
    var_20 = module_0.Assign(*var_16, **var_19)
    var_21 = [var_20]
    var_22 = 'z'
    var_23 = []
    var_24 = 'id'
    var_25 = {var_24: var_22}
    var_26 = module_0.Name(*var_23, **var_25)
    var_27 = [var_26]
    var_28 = 3
    var_29 = []
    var_30 = 'value'
    var_31 = {var_30: var_28}
    var_32 = module_0.Constant(*var_29, **var_31)
    var_33 = []
    var_34 = 'targets'
    var_35 = 'value'
    var_36 = {var_34: var_27, var_35: var_32}
    var_37 = module_0.Assign(*var_33, **var_36)
    var_38 = [var_37]
    var_39 = []
    var_40 = 'test'
    var_41 = 'body'
    var_42 = 'orelse'
    var_43 = {var_40: var_4, var_41: var_21, var_42: var_38}
    var_44 = module_0.If(*var_39, **var_43)
    var_45 = [var_44]
    var_46 = module_1.walk_body(var_45)
    var_47 = list(var_46)
    var_48 = []
    var_49 = 'id'
    var_50 = {var_49: var_5}
    var_51 = module_0.Name(*var_48, **var_50)
    var_52 = [var_51]
    var_53 = []
    var_54 = 'value'
    var_55 = {var_54: var_11}
    var_56 = module_0.Constant(*var_53, **var_55)
    var_57 = []
    var_58 = 'targets'
    var_59 = 'value'
    var_60 = {var_58: var_52, var_59: var_56}
    var_61 = module_0.Assign(*var_57, **var_60)
    var_62 = []
    var_63 = 'id'
    var_64 = {var_63: var_22}
    var_65 = module_0.Name(*var_62, **var_64)
    var_66 = [var_65]
    var_67 = []
    var_68 = 'value'
    var_69 = {var_68: var_28}
    var_70 = module_0.Constant(*var_67, **var_69)
    var_71 = []
    var_72 = 'targets'
    var_73 = 'value'
    var_74 = {var_72: var_66, var_73: var_70}
    var_75 = module_0.Assign(*var_71, **var_74)
    var_76 = [var_61, var_75]
    var_77 = bool(var_47 == var_76)
    assert var_77 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = 'body'
    var_22 = 'handlers'
    var_23 = 'orelse'
    var_24 = 'finalbody'
    var_25 = {var_21: var_16, var_22: var_17, var_23: var_18, var_24: var_19}
    var_26 = module_0.Try(*var_20, **var_25)
    var_27 = [var_26]
    var_28 = module_1.walk_body(var_27)
    var_29 = list(var_28)
    var_30 = []
    var_31 = 'id'
    var_32 = {var_31: var_0}
    var_33 = module_0.Name(*var_30, **var_32)
    var_34 = [var_33]
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_6}
    var_38 = module_0.Constant(*var_35, **var_37)
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_34, var_41: var_38}
    var_43 = module_0.Assign(*var_39, **var_42)
    var_44 = [var_43]
    var_45 = bool(var_29 == var_44)
    assert var_45 is True

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = 'b'
    var_18 = []
    var_19 = 'id'
    var_20 = {var_19: var_17}
    var_21 = module_0.Name(*var_18, **var_20)
    var_22 = [var_21]
    var_23 = 2
    var_24 = []
    var_25 = 'value'
    var_26 = {var_25: var_23}
    var_27 = module_0.Constant(*var_24, **var_26)
    var_28 = []
    var_29 = 'targets'
    var_30 = 'value'
    var_31 = {var_29: var_22, var_30: var_27}
    var_32 = module_0.Assign(*var_28, **var_31)
    var_33 = [var_32]
    var_34 = []
    var_35 = []
    var_36 = []
    var_37 = 'id'
    var_38 = {var_37: var_0}
    var_39 = module_0.Name(*var_36, **var_38)
    var_40 = [var_39]
    var_41 = []
    var_42 = 'value'
    var_43 = {var_42: var_6}
    var_44 = module_0.Constant(*var_41, **var_43)
    var_45 = []
    var_46 = 'targets'
    var_47 = 'value'
    var_48 = {var_46: var_40, var_47: var_44}
    var_49 = module_0.Assign(*var_45, **var_48)
    var_50 = []
    var_51 = 'id'
    var_52 = {var_51: var_17}
    var_53 = module_0.Name(*var_50, **var_52)
    var_54 = [var_53]
    var_55 = []
    var_56 = 'value'
    var_57 = {var_56: var_23}
    var_58 = module_0.Constant(*var_55, **var_57)
    var_59 = []
    var_60 = 'targets'
    var_61 = 'value'
    var_62 = {var_60: var_54, var_61: var_58}
    var_63 = module_0.Assign(*var_59, **var_62)
    var_64 = [var_49, var_63]

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = 'b'
    var_19 = []
    var_20 = 'id'
    var_21 = {var_20: var_18}
    var_22 = module_0.Name(*var_19, **var_21)
    var_23 = [var_22]
    var_24 = 2
    var_25 = []
    var_26 = 'value'
    var_27 = {var_26: var_24}
    var_28 = module_0.Constant(*var_25, **var_27)
    var_29 = []
    var_30 = 'targets'
    var_31 = 'value'
    var_32 = {var_30: var_23, var_31: var_28}
    var_33 = module_0.Assign(*var_29, **var_32)
    var_34 = [var_33]
    var_35 = 'c'
    var_36 = []
    var_37 = 'id'
    var_38 = {var_37: var_35}
    var_39 = module_0.Name(*var_36, **var_38)
    var_40 = [var_39]
    var_41 = 3
    var_42 = []
    var_43 = 'value'
    var_44 = {var_43: var_41}
    var_45 = module_0.Constant(*var_42, **var_44)
    var_46 = []
    var_47 = 'targets'
    var_48 = 'value'
    var_49 = {var_47: var_40, var_48: var_45}
    var_50 = module_0.Assign(*var_46, **var_49)
    var_51 = [var_50]
    var_52 = []
    var_53 = 'body'
    var_54 = 'handlers'
    var_55 = 'orelse'
    var_56 = 'finalbody'
    var_57 = {var_53: var_16, var_54: var_17, var_55: var_34, var_56: var_51}
    var_58 = module_0.Try(*var_52, **var_57)
    var_59 = [var_58]
    var_60 = module_1.walk_body(var_59)
    var_61 = list(var_60)
    var_62 = []
    var_63 = 'id'
    var_64 = {var_63: var_0}
    var_65 = module_0.Name(*var_62, **var_64)
    var_66 = [var_65]
    var_67 = []
    var_68 = 'value'
    var_69 = {var_68: var_6}
    var_70 = module_0.Constant(*var_67, **var_69)
    var_71 = []
    var_72 = 'targets'
    var_73 = 'value'
    var_74 = {var_72: var_66, var_73: var_70}
    var_75 = module_0.Assign(*var_71, **var_74)
    var_76 = []
    var_77 = 'id'
    var_78 = {var_77: var_18}
    var_79 = module_0.Name(*var_76, **var_78)
    var_80 = [var_79]
    var_81 = []
    var_82 = 'value'
    var_83 = {var_82: var_24}
    var_84 = module_0.Constant(*var_81, **var_83)
    var_85 = []
    var_86 = 'targets'
    var_87 = 'value'
    var_88 = {var_86: var_80, var_87: var_84}
    var_89 = module_0.Assign(*var_85, **var_88)
    var_90 = []
    var_91 = 'id'
    var_92 = {var_91: var_35}
    var_93 = module_0.Name(*var_90, **var_92)
    var_94 = [var_93]
    var_95 = []
    var_96 = 'value'
    var_97 = {var_96: var_41}
    var_98 = module_0.Constant(*var_95, **var_97)
    var_99 = []
    var_100 = 'targets'
    var_101 = 'value'
    var_102 = {var_100: var_94, var_101: var_98}
    var_103 = module_0.Assign(*var_99, **var_102)
    var_104 = [var_75, var_89, var_103]
    var_105 = bool(var_61 == var_104)
    assert var_105 is True

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_5, var_13: var_10}
    var_15 = module_0.Assign(*var_11, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = {var_17: var_0}
    var_19 = module_0.Name(*var_16, **var_18)
    var_20 = 'y'
    var_21 = []
    var_22 = 'id'
    var_23 = {var_22: var_20}
    var_24 = module_0.Name(*var_21, **var_23)
    var_25 = [var_24]
    var_26 = 2
    var_27 = []
    var_28 = 'value'
    var_29 = {var_28: var_26}
    var_30 = module_0.Constant(*var_27, **var_29)
    var_31 = []
    var_32 = 'targets'
    var_33 = 'value'
    var_34 = {var_32: var_25, var_33: var_30}
    var_35 = module_0.Assign(*var_31, **var_34)
    var_36 = [var_35]
    var_37 = 'z'
    var_38 = []
    var_39 = 'id'
    var_40 = {var_39: var_37}
    var_41 = module_0.Name(*var_38, **var_40)
    var_42 = [var_41]
    var_43 = 3
    var_44 = []
    var_45 = 'value'
    var_46 = {var_45: var_43}
    var_47 = module_0.Constant(*var_44, **var_46)
    var_48 = []
    var_49 = 'targets'
    var_50 = 'value'
    var_51 = {var_49: var_42, var_50: var_47}
    var_52 = module_0.Assign(*var_48, **var_51)
    var_53 = [var_52]
    var_54 = []
    var_55 = 'test'
    var_56 = 'body'
    var_57 = 'orelse'
    var_58 = {var_55: var_19, var_56: var_36, var_57: var_53}
    var_59 = module_0.If(*var_54, **var_58)
    var_60 = 'a'
    var_61 = []
    var_62 = 'id'
    var_63 = {var_62: var_60}
    var_64 = module_0.Name(*var_61, **var_63)
    var_65 = [var_64]
    var_66 = 4
    var_67 = []
    var_68 = 'value'
    var_69 = {var_68: var_66}
    var_70 = module_0.Constant(*var_67, **var_69)
    var_71 = []
    var_72 = 'targets'
    var_73 = 'value'
    var_74 = {var_72: var_65, var_73: var_70}
    var_75 = module_0.Assign(*var_71, **var_74)
    var_76 = [var_75]
    var_77 = 'b'
    var_78 = []
    var_79 = 'id'
    var_80 = {var_79: var_77}
    var_81 = module_0.Name(*var_78, **var_80)
    var_82 = [var_81]
    var_83 = 5
    var_84 = []
    var_85 = 'value'
    var_86 = {var_85: var_83}
    var_87 = module_0.Constant(*var_84, **var_86)
    var_88 = []
    var_89 = 'targets'
    var_90 = 'value'
    var_91 = {var_89: var_82, var_90: var_87}
    var_92 = module_0.Assign(*var_88, **var_91)
    var_93 = [var_92]
    var_94 = 'c'
    var_95 = []
    var_96 = 'id'
    var_97 = {var_96: var_94}
    var_98 = module_0.Name(*var_95, **var_97)
    var_99 = [var_98]
    var_100 = 6
    var_101 = []
    var_102 = 'value'
    var_103 = {var_102: var_100}
    var_104 = module_0.Constant(*var_101, **var_103)
    var_105 = []
    var_106 = 'targets'
    var_107 = 'value'
    var_108 = {var_106: var_99, var_107: var_104}
    var_109 = module_0.Assign(*var_105, **var_108)
    var_110 = [var_109]
    var_111 = 'd'
    var_112 = []
    var_113 = 'id'
    var_114 = {var_113: var_111}
    var_115 = module_0.Name(*var_112, **var_114)
    var_116 = [var_115]
    var_117 = 7
    var_118 = []
    var_119 = 'value'
    var_120 = {var_119: var_117}
    var_121 = module_0.Constant(*var_118, **var_120)
    var_122 = []
    var_123 = 'targets'
    var_124 = 'value'
    var_125 = {var_123: var_116, var_124: var_121}
    var_126 = module_0.Assign(*var_122, **var_125)
    var_127 = [var_126]
    var_128 = []
    var_129 = 'id'
    var_130 = {var_129: var_0}
    var_131 = module_0.Name(*var_128, **var_130)
    var_132 = [var_131]
    var_133 = []
    var_134 = 'value'
    var_135 = {var_134: var_6}
    var_136 = module_0.Constant(*var_133, **var_135)
    var_137 = []
    var_138 = 'targets'
    var_139 = 'value'
    var_140 = {var_138: var_132, var_139: var_136}
    var_141 = module_0.Assign(*var_137, **var_140)
    var_142 = []
    var_143 = 'id'
    var_144 = {var_143: var_20}
    var_145 = module_0.Name(*var_142, **var_144)
    var_146 = [var_145]
    var_147 = []
    var_148 = 'value'
    var_149 = {var_148: var_26}
    var_150 = module_0.Constant(*var_147, **var_149)
    var_151 = []
    var_152 = 'targets'
    var_153 = 'value'
    var_154 = {var_152: var_146, var_153: var_150}
    var_155 = module_0.Assign(*var_151, **var_154)
    var_156 = []
    var_157 = 'id'
    var_158 = {var_157: var_37}
    var_159 = module_0.Name(*var_156, **var_158)
    var_160 = [var_159]
    var_161 = []
    var_162 = 'value'
    var_163 = {var_162: var_43}
    var_164 = module_0.Constant(*var_161, **var_163)
    var_165 = []
    var_166 = 'targets'
    var_167 = 'value'
    var_168 = {var_166: var_160, var_167: var_164}
    var_169 = module_0.Assign(*var_165, **var_168)
    var_170 = []
    var_171 = 'id'
    var_172 = {var_171: var_60}
    var_173 = module_0.Name(*var_170, **var_172)
    var_174 = [var_173]
    var_175 = []
    var_176 = 'value'
    var_177 = {var_176: var_66}
    var_178 = module_0.Constant(*var_175, **var_177)
    var_179 = []
    var_180 = 'targets'
    var_181 = 'value'
    var_182 = {var_180: var_174, var_181: var_178}
    var_183 = module_0.Assign(*var_179, **var_182)
    var_184 = []
    var_185 = 'id'
    var_186 = {var_185: var_77}
    var_187 = module_0.Name(*var_184, **var_186)
    var_188 = [var_187]
    var_189 = []
    var_190 = 'value'
    var_191 = {var_190: var_83}
    var_192 = module_0.Constant(*var_189, **var_191)
    var_193 = []
    var_194 = 'targets'
    var_195 = 'value'
    var_196 = {var_194: var_188, var_195: var_192}
    var_197 = module_0.Assign(*var_193, **var_196)
    var_198 = []
    var_199 = 'id'
    var_200 = {var_199: var_94}
    var_201 = module_0.Name(*var_198, **var_200)
    var_202 = [var_201]
    var_203 = []
    var_204 = 'value'
    var_205 = {var_204: var_100}
    var_206 = module_0.Constant(*var_203, **var_205)
    var_207 = []
    var_208 = 'targets'
    var_209 = 'value'
    var_210 = {var_208: var_202, var_209: var_206}
    var_211 = module_0.Assign(*var_207, **var_210)
    var_212 = []
    var_213 = 'id'
    var_214 = {var_213: var_111}
    var_215 = module_0.Name(*var_212, **var_214)
    var_216 = [var_215]
    var_217 = []
    var_218 = 'value'
    var_219 = {var_218: var_117}
    var_220 = module_0.Constant(*var_217, **var_219)
    var_221 = []
    var_222 = 'targets'
    var_223 = 'value'
    var_224 = {var_222: var_216, var_223: var_220}
    var_225 = module_0.Assign(*var_221, **var_224)
    var_226 = [var_141, var_155, var_169, var_183, var_197, var_211, var_225]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 15/17 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 15/17 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 13/15 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 13/15 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 19/21 statements.
# Partially parsed test_func_api_with_self. Retrieved 14/16 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 13/15 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 13/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'b'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'root'
    var_17 = 'func'
    var_18 = False
    var_19 = '| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n'
    var_20 = var_0.doc['root.func']
    var_21 = bool(var_0.doc['root.func'] == var_19)
    assert var_21 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'a'
    var_5 = None
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'b'
    var_10 = [var_9, var_5]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = []
    var_15 = []
    var_16 = 'root'
    var_17 = 'func'
    var_18 = False
    var_19 = '| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n'
    var_20 = var_0.doc['root.func']
    var_21 = bool(var_0.doc['root.func'] == var_19)
    assert var_21 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = 'args'
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = []
    var_12 = 'root'
    var_13 = 'func'
    var_14 = False
    var_15 = '| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'
    var_16 = var_0.doc['root.func']
    var_17 = bool(var_0.doc['root.func'] == var_15)
    assert var_17 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = []
    var_12 = 'root'
    var_13 = 'func'
    var_14 = False
    var_15 = '| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'
    var_16 = var_0.doc['root.func']
    var_17 = bool(var_0.doc['root.func'] == var_15)
    assert var_17 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = 'b'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = [var_6, var_10]
    var_12 = 1
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Constant(*var_13, **var_14)
    var_16 = 2
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Constant(*var_17, **var_18)
    var_20 = [var_15, var_19]
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'root'
    var_25 = 'func'
    var_26 = False
    var_27 = '| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n| `1` | `2` |  |\n\n'
    var_28 = var_0.doc['root.func']
    var_29 = bool(var_0.doc['root.func'] == var_27)
    assert var_29 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'root'
    var_13 = 'func'
    var_14 = True
    var_15 = False
    var_16 = '| self | return |\n|:---:|:---:|\n| `Self` | `Any` |\n\n'
    var_17 = var_0.doc['root.func']
    var_18 = bool(var_0.doc['root.func'] == var_16)
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'cls'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'root'
    var_13 = 'func'
    var_14 = True
    var_15 = '| cls | return |\n|:---:|:---:|\n| `type[Self]` | `Any` |\n\n'
    var_16 = var_0.doc['root.func']
    var_17 = bool(var_0.doc['root.func'] == var_15)
    assert var_17 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = []
    var_8 = 'root'
    var_9 = 'func'
    var_10 = 'int'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Constant(*var_11, **var_12)
    var_14 = False
    var_15 = '| return |\n|:---:|\n| `int` |\n\n'
    var_16 = var_0.doc['root.func']
    var_17 = bool(var_0.doc['root.func'] == var_15)
    assert var_17 is True



# Parsed testcases at query #24
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'class_name'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc['root.class_name'])
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_docstring_updates_docstring_when_doc_exists. Retrieved 14/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod'
    var_2 = '# Module `pkg.submod`'
    var_3 = 'MockModule'
    var_4 = ()
    var_5 = 'submod'
    var_6 = '__doc__'
    var_7 = 'Test doc'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = type(var_3, var_4, var_9)
    var_11 = var_10()
    var_12 = 'pkg'
    var_13 = var_0.load_docstring(var_12, var_11)
    var_14 = 'pkg.submod'
    var_15 = bool('pkg.submod' in var_0.docstring)
    assert var_15 is True
    var_16 = var_0.docstring['pkg.submod']
    assert var_16 == 'Test doc'



# Parsed testcases at query #26
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 123
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'root'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = '__all__'
    var_20 = bool('__all__' not in var_0.imp)
    assert var_20 is True



# Parsed testcases at query #27
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
    var_4 = var_0.alias['test.module.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = var_0.alias['test.module.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['test.module.submodule.path']
    assert var_6 == 'test.module.os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = 0
    var_6 = var_0.alias['test.module.ospath']
    assert var_6 == 'os.path'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = 'int'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    var_11 = 1
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'target'
    var_18 = 'annotation'
    var_19 = 'value'
    var_20 = {var_17: var_5, var_18: var_10, var_19: var_15}
    var_21 = module_1.AnnAssign(*var_16, **var_20)
    var_22 = 'module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = var_0.alias['module.var']
    assert var_24 == '1'
    var_25 = var_0.const['module.VAR']
    assert var_25 == 'int'
    var_26 = var_0.root['module.VAR']
    assert var_26 == 'module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11}
    var_16 = module_1.Assign(*var_12, **var_15)
    var_17 = 'module'
    var_18 = var_0.globals(var_17, var_16)
    var_19 = var_0.alias['module.var']
    assert var_19 == '1'
    var_20 = var_0.const['module.VAR']
    assert var_20 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'func'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.List(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'module'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = 'module.func'
    var_25 = bool('module.func' in var_0.imp['module'])
    assert var_25 is True



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.function'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private_function'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = '_local_function'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.__init__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module._private.__magic__'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True

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



# Parsed testcases at query #3
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
    var_4 = var_0.alias['test.module.os']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = var_0.alias['test.module.operating_system']
    assert var_4 == 'os'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.submodule'
    var_2 = 'os'
    var_3 = 'path'
    var_4 = None
    var_5 = 1
    var_6 = var_0.alias['test.module.submodule.path']
    assert var_6 == 'test.module.os.path'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = 'j'
    var_5 = 0
    var_6 = var_0.alias['test.module.j']
    assert var_6 == 'os.path.join'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_imports_with_import_node. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'import os'
    var_2 = True
    var_3 = module_1.parse(var_1, type_comments=var_2)
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = 'test'
    var_7 = var_0.imports(var_6, var_5)
    var_8 = var_0.alias['test.os']
    assert var_8 == 'os'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_visit_Constant_with_valid_name. Retrieved 8/9 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 123
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'invalid syntax'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_2.visit_Constant(var_7)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'test.alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'name'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Constant(*var_6, **var_8)
    var_10 = var_4.visit_Constant(var_9)
    var_11 = var_10.id
    assert var_11 == 'alias'



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'Title'
    var_1 = 'Item1'
    var_2 = 'Item2'
    var_3 = [var_1, var_2]
    var_4 = [var_0]
    var_5 = module_0.table(*var_4, items=var_3)
    assert var_5 == '| Title |\n|:-----:|\n| Item1 |\n| Item2 |\n\n'

import apimd.parser as module_0

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
    var_10 = module_0.table(*var_9, items=var_8)
    assert var_10 == '| A | B |\n|:---:|:---:|\n| 1 | 2 |\n| 3 | 4 |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'X'
    var_1 = 'Y'
    var_2 = 'Single'
    var_3 = 'A'
    var_4 = 'B'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_0, var_1]
    var_8 = module_0.table(*var_7, items=var_6)
    assert var_8 == '| X | Y |\n|:---:|:---:|\n| Single |   |\n| A | B |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'A'
    var_2 = 'B'
    var_3 = [var_1, var_2]
    var_4 = [var_3]
    var_5 = [var_0]
    var_6 = module_0.table(*var_5, items=var_4)
    assert var_6 == '|   |   |\n|:---:|:---:|\n| A | B |\n\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'VeryLongTitle'
    var_1 = 'AnotherLongTitle'
    var_2 = 'Short'
    var_3 = 'Data'
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = [var_0, var_1]
    var_7 = module_0.table(*var_6, items=var_5)
    assert var_7 == '| VeryLongTitle | AnotherLongTitle |\n|:--------------:|:----------------:|\n| Short | Data |\n\n'



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'x: int = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = "y: str = 'hello'"
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = 'z = 3.14'
    var_15 = module_1.parse(var_14, mode=var_4)
    var_16 = var_15.body
    var_17 = [var_10, var_13, var_16]
    var_18 = var_0.class_api(var_1, var_2, var_7, var_17)
    var_19 = 'Bases'
    var_20 = bool('Bases' in var_0.doc[var_2])
    assert var_20 is True
    var_21 = 'Members'
    var_22 = bool('Members' in var_0.doc[var_2])
    assert var_22 is True
    var_23 = 'Type'
    var_24 = bool('Type' in var_0.doc[var_2])
    assert var_24 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'A = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = 'B = 2'
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = [var_10, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_7, var_14)
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc[var_2])
    assert var_17 is True
    var_18 = 'A'
    var_19 = bool('A' in var_0.doc[var_2])
    assert var_19 is True
    var_20 = 'B'
    var_21 = bool('B' in var_0.doc[var_2])
    assert var_21 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.EmptyClass'
    var_3 = []
    var_4 = []
    var_5 = var_0.class_api(var_1, var_2, var_3, var_4)
    var_6 = 'Bases'
    var_7 = bool('Bases' not in var_0.doc[var_2])
    assert var_7 is True
    var_8 = 'Members'
    var_9 = bool('Members' not in var_0.doc[var_2])
    assert var_9 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = '_private: int = 1'
    var_5 = 'eval'
    var_6 = module_1.parse(var_4, mode=var_5)
    var_7 = var_6.body
    var_8 = "public: str = 'hello'"
    var_9 = module_1.parse(var_8, mode=var_5)
    var_10 = var_9.body
    var_11 = [var_7, var_10]
    var_12 = var_0.class_api(var_1, var_2, var_3, var_11)
    var_13 = 'Members'
    var_14 = bool('Members' in var_0.doc[var_2])
    assert var_14 is True
    var_15 = '_private'
    var_16 = bool('_private' not in var_0.doc[var_2])
    assert var_16 is True
    var_17 = 'public'
    var_18 = bool('public' in var_0.doc[var_2])
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'x: int = 1'
    var_5 = 'eval'
    var_6 = module_1.parse(var_4, mode=var_5)
    var_7 = var_6.body
    var_8 = 'del x'
    var_9 = module_1.parse(var_8, mode=var_5)
    var_10 = var_9.body
    var_11 = [var_7, var_10]
    var_12 = var_0.class_api(var_1, var_2, var_3, var_11)
    var_13 = 'x'
    var_14 = bool('x' not in var_0.doc[var_2])
    assert var_14 is True



# Parsed testcases at query #8
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'attr1'
    var_5 = []
    var_6 = 'id'
    var_7 = {var_6: var_4}
    var_8 = module_1.Name(*var_5, **var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = 'targets'
    var_12 = {var_11: var_9}
    var_13 = module_1.Delete(*var_10, **var_12)
    var_14 = 'attr2'
    var_15 = []
    var_16 = 'id'
    var_17 = {var_16: var_14}
    var_18 = module_1.Name(*var_15, **var_17)
    var_19 = [var_18]
    var_20 = []
    var_21 = 'targets'
    var_22 = {var_21: var_19}
    var_23 = module_1.Delete(*var_20, **var_22)
    var_24 = [var_13, var_23]
    var_25 = var_0.class_api(var_1, var_2, var_3, var_24)
    var_26 = 'attr1'
    var_27 = bool('attr1' not in var_0.doc[var_2])
    assert var_27 is True
    var_28 = 'attr2'
    var_29 = bool('attr2' not in var_0.doc[var_2])
    assert var_29 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 'This is a regular line.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a regular line.'

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('hello')\n```"

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Line 1\nLine 2\nLine 3'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'Line 1\nLine 2\nLine 3'

import apimd.parser as module_0

def test_case_0():
    var_0 = "Regular line\n>>> print('hello')\nAnother regular line"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line\n```python\n>>> print('hello')\n```\nAnother regular line"

import apimd.parser as module_0

def test_case_0():
    var_0 = "Regular line\n>>> print('hello')\n>>> print('world')\nAnother regular line"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line\n```python\n>>> print('hello')\n>>> print('world')\n```\nAnother regular line"

import apimd.parser as module_0

def test_case_0():
    var_0 = "Regular line\n>>> print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line\n```python\n>>> print('hello')\n```"

import apimd.parser as module_0

def test_case_0():
    var_0 = ">>> print('hello')\nRegular line"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('hello')\n```\nRegular line"

import apimd.parser as module_0

def test_case_0():
    var_0 = "Regular line 1\n>>> print('hello')\nRegular line 2"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line 1\n```python\n>>> print('hello')\n```\nRegular line 2"



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 9/10 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_2, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = var_3.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.Name'
    var_2 = 'alias.Name'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'Name'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = var_4.visit_Name(var_13)
    var_15 = var_14.id
    assert var_15 == 'alias.Name'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'Name'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = var_2.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'Name'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'root'
    var_1 = 'root.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'T'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = []
    var_10 = 'id'
    var_11 = 'ctx'
    var_12 = {var_10: var_5, var_11: var_8}
    var_13 = module_1.Name(*var_9, **var_12)
    var_14 = var_4.visit_Name(var_13)
    var_15 = var_14.id
    assert var_15 == 'T'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_const_type_call. Retrieved 14/18 statements.


import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'int'
    var_5 = 'hello'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Constant(*var_6, **var_7)
    var_9 = module_1.const_type(var_8)
    assert var_9 == 'str'
    var_10 = 3.14
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Constant(*var_11, **var_12)
    var_14 = module_1.const_type(var_13)
    assert var_14 == 'float'
    var_15 = True
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Constant(*var_16, **var_17)
    var_19 = module_1.const_type(var_18)
    assert var_19 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Tuple(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'tuple[int, int]'
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = 'b'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.Constant(*var_18, **var_19)
    var_21 = [var_16, var_20]
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_0.Tuple(*var_22, **var_23)
    var_25 = module_1.const_type(var_24)
    assert var_25 == 'tuple[str, str]'
    var_26 = [var_0]
    var_27 = {}
    var_28 = module_0.Constant(*var_26, **var_27)
    var_29 = [var_13]
    var_30 = {}
    var_31 = module_0.Constant(*var_29, **var_30)
    var_32 = [var_28, var_31]
    var_33 = [var_32]
    var_34 = {}
    var_35 = module_0.Tuple(*var_33, **var_34)
    var_36 = module_1.const_type(var_35)
    assert var_36 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.List(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'list[int]'
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = 'b'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.Constant(*var_18, **var_19)
    var_21 = [var_16, var_20]
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_0.List(*var_22, **var_23)
    var_25 = module_1.const_type(var_24)
    assert var_25 == 'list[str]'
    var_26 = [var_0]
    var_27 = {}
    var_28 = module_0.Constant(*var_26, **var_27)
    var_29 = [var_13]
    var_30 = {}
    var_31 = module_0.Constant(*var_29, **var_30)
    var_32 = [var_28, var_31]
    var_33 = [var_32]
    var_34 = {}
    var_35 = module_0.List(*var_33, **var_34)
    var_36 = module_1.const_type(var_35)
    assert var_36 == 'list[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Set(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'set[int]'
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = 'b'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.Constant(*var_18, **var_19)
    var_21 = [var_16, var_20]
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_0.Set(*var_22, **var_23)
    var_25 = module_1.const_type(var_24)
    assert var_25 == 'set[str]'
    var_26 = [var_0]
    var_27 = {}
    var_28 = module_0.Constant(*var_26, **var_27)
    var_29 = [var_13]
    var_30 = {}
    var_31 = module_0.Constant(*var_29, **var_30)
    var_32 = [var_28, var_31]
    var_33 = [var_32]
    var_34 = {}
    var_35 = module_0.Set(*var_33, **var_34)
    var_36 = module_1.const_type(var_35)
    assert var_36 == 'set[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Constant(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = [var_4, var_9]
    var_11 = {}
    var_12 = module_0.Dict(*var_10, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'dict[int, str]'
    var_14 = [var_5]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = [var_0]
    var_19 = {}
    var_20 = module_0.Constant(*var_18, **var_19)
    var_21 = [var_20]
    var_22 = [var_17, var_21]
    var_23 = {}
    var_24 = module_0.Dict(*var_22, **var_23)
    var_25 = module_1.const_type(var_24)
    assert var_25 == 'dict[str, int]'
    var_26 = [var_0]
    var_27 = {}
    var_28 = module_0.Constant(*var_26, **var_27)
    var_29 = [var_5]
    var_30 = {}
    var_31 = module_0.Constant(*var_29, **var_30)
    var_32 = [var_28, var_31]
    var_33 = 'b'
    var_34 = [var_33]
    var_35 = {}
    var_36 = module_0.Constant(*var_34, **var_35)
    var_37 = 2
    var_38 = [var_37]
    var_39 = {}
    var_40 = module_0.Constant(*var_38, **var_39)
    var_41 = [var_36, var_40]
    var_42 = [var_32, var_41]
    var_43 = {}
    var_44 = module_0.Dict(*var_42, **var_43)
    var_45 = module_1.const_type(var_44)
    assert var_45 == 'dict[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Name(*var_1, **var_2)
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.Call(*var_5, **var_6)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'int'
    var_9 = 'str'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Name(*var_10, **var_11)
    var_13 = []
    var_14 = [var_12, var_13]
    var_15 = {}
    var_16 = module_0.Call(*var_14, **var_15)
    var_17 = module_1.const_type(var_16)
    assert var_17 == 'str'
    var_18 = 1
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_0.Constant(*var_19, **var_20)
    var_22 = 'real'
    var_23 = []

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'unknown'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Name(*var_1, **var_2)
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.Call(*var_5, **var_6)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'Any'
    var_9 = 'x'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Name(*var_10, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'Any'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_docstring_with_valid_module. Retrieved 16/20 statements.
# Partially parsed test_load_docstring_with_none_doc. Retrieved 15/19 statements.
# Partially parsed test_load_docstring_with_nested_attribute. Retrieved 19/23 statements.
# Partially parsed test_load_docstring_with_missing_attribute. Retrieved 13/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'Module `pkg.module`'
    var_4 = 'Function `func`'
    var_5 = 'module'
    var_6 = ()
    var_7 = '__doc__'
    var_8 = 'func'
    var_9 = 'Module doc'
    var_10 = None
    var_11 = lambda : var_10
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = type(var_5, var_6, var_12)
    var_14 = var_13()
    var_15 = var_0.load_docstring(var_1, var_14)
    var_16 = var_0.docstring['pkg.module']
    assert var_16 == 'Module doc'
    var_17 = var_0.docstring['pkg.module.func']
    assert var_17 == 'Function doc'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.func'
    var_3 = 'Module `pkg.module`'
    var_4 = 'Function `func`'
    var_5 = 'module'
    var_6 = ()
    var_7 = '__doc__'
    var_8 = 'func'
    var_9 = None
    var_10 = lambda : var_9
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = type(var_5, var_6, var_11)
    var_13 = var_12()
    var_14 = var_0.load_docstring(var_1, var_13)
    var_15 = 'pkg.module'
    var_16 = bool('pkg.module' not in var_0.docstring)
    assert var_16 is True
    var_17 = 'pkg.module.func'
    var_18 = bool('pkg.module.func' not in var_0.docstring)
    assert var_18 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.Class.attr'
    var_3 = 'Module `pkg.module`'
    var_4 = 'Attribute `attr`'
    var_5 = 'module'
    var_6 = ()
    var_7 = '__doc__'
    var_8 = 'Class'
    var_9 = 'Module doc'
    var_10 = ()
    var_11 = 'attr'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = type(var_8, var_10, var_13)
    var_15 = {var_7: var_9, var_8: var_14}
    var_16 = type(var_5, var_6, var_15)
    var_17 = var_16()
    var_18 = var_0.load_docstring(var_1, var_17)
    var_19 = var_0.docstring['pkg.module']
    assert var_19 == 'Module doc'
    var_20 = var_0.docstring['pkg.module.Class.attr']
    assert var_20 == 'Attribute doc'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.module'
    var_2 = 'pkg.module.missing'
    var_3 = 'Module `pkg.module`'
    var_4 = 'Missing `missing`'
    var_5 = 'module'
    var_6 = ()
    var_7 = '__doc__'
    var_8 = 'Module doc'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = var_0.load_docstring(var_1, var_11)
    var_13 = var_0.docstring['pkg.module']
    assert var_13 == 'Module doc'
    var_14 = 'pkg.module.missing'
    var_15 = bool('pkg.module.missing' not in var_0.docstring)
    assert var_15 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_public_with_all_list. Retrieved 10/13 statements.
# Partially parsed test_is_public_without_all_list. Retrieved 8/11 statements.
# Partially parsed test_is_public_private_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_magic_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_nested_module. Retrieved 7/10 statements.
# Partially parsed test_is_public_not_in_all. Retrieved 7/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = 'mod2'
    var_4 = {var_2, var_3}
    var_5 = 'pkg.mod1'
    var_6 = 'pkg.mod2'
    var_7 = ''
    var_8 = var_0.is_public(var_5)
    assert var_8 is True
    var_9 = var_0.is_public(var_6)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.mod1'
    var_4 = 'pkg.mod2'
    var_5 = ''
    var_6 = var_0.is_public(var_3)
    assert var_6 is True
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg._private'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = set()
    var_3 = 'pkg.__init__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = {var_2}
    var_4 = 'pkg.mod1.submod'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'mod1'
    var_3 = {var_2}
    var_4 = 'pkg.mod2'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_public_with_public_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_private_name. Retrieved 5/7 statements.
# Partially parsed test_is_public_with_magic_name. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_nested_public_name. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_nested_private_name. Retrieved 6/8 statements.
# Partially parsed test_is_public_with_all_listed_name. Retrieved 7/10 statements.
# Partially parsed test_is_public_with_parent_in_all. Retrieved 8/11 statements.
# Partially parsed test_is_public_with_empty_all. Retrieved 6/9 statements.
# Partially parsed test_is_public_with_no_doc. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'public_name'
    var_3 = {var_2}
    var_4 = ''
    var_5 = var_0.is_public(var_2)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = '_private_name'
    var_4 = var_0.is_public(var_3)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = '__magic__'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'nested'
    var_3 = {var_2}
    var_4 = 'root.nested.public_name'
    var_5 = 'root.nested'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'root.nested._private_name'
    var_4 = 'root.nested'
    var_5 = var_0.is_public(var_3)
    assert var_5 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'listed_name'
    var_3 = {var_2}
    var_4 = 'root.listed_name'
    var_5 = ''
    var_6 = var_0.is_public(var_4)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'root.parent.child'
    var_5 = 'root.parent'
    var_6 = ''
    var_7 = var_0.is_public(var_4)
    assert var_7 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = set()
    var_3 = 'public_name'
    var_4 = ''
    var_5 = var_0.is_public(var_3)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'no_doc'
    var_3 = {var_2}
    var_4 = var_0.is_public(var_2)
    assert var_4 is False



# Parsed testcases at query #15
#--------------------------




import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'int'
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'float'
    var_9 = 'hello'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Constant(*var_10, **var_11)
    var_13 = module_1.const_type(var_12)
    assert var_13 == 'str'
    var_14 = True
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_0.Constant(*var_15, **var_16)
    var_18 = module_1.const_type(var_17)
    assert var_18 == 'bool'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Tuple(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'tuple[int, int]'
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Constant(*var_13, **var_14)
    var_16 = [var_4]
    var_17 = {}
    var_18 = module_0.Constant(*var_16, **var_17)
    var_19 = [var_15, var_18]
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_0.Tuple(*var_20, **var_21)
    var_23 = module_1.const_type(var_22)
    assert var_23 == 'tuple[Any, Any]'
    var_24 = [var_0]
    var_25 = {}
    var_26 = module_0.Constant(*var_24, **var_25)
    var_27 = 'hello'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_0.Constant(*var_28, **var_29)
    var_31 = [var_26, var_30]
    var_32 = [var_31]
    var_33 = {}
    var_34 = module_0.Tuple(*var_32, **var_33)
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'tuple[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.List(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'list[int, int]'
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Constant(*var_13, **var_14)
    var_16 = [var_4]
    var_17 = {}
    var_18 = module_0.Constant(*var_16, **var_17)
    var_19 = [var_15, var_18]
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_0.List(*var_20, **var_21)
    var_23 = module_1.const_type(var_22)
    assert var_23 == 'list[Any, Any]'
    var_24 = [var_0]
    var_25 = {}
    var_26 = module_0.Constant(*var_24, **var_25)
    var_27 = 'hello'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_0.Constant(*var_28, **var_29)
    var_31 = [var_26, var_30]
    var_32 = [var_31]
    var_33 = {}
    var_34 = module_0.List(*var_32, **var_33)
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'list[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Set(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'set[int, int]'
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Constant(*var_13, **var_14)
    var_16 = [var_4]
    var_17 = {}
    var_18 = module_0.Constant(*var_16, **var_17)
    var_19 = [var_15, var_18]
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_0.Set(*var_20, **var_21)
    var_23 = module_1.const_type(var_22)
    assert var_23 == 'set[Any, Any]'
    var_24 = [var_0]
    var_25 = {}
    var_26 = module_0.Constant(*var_24, **var_25)
    var_27 = 'hello'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_0.Constant(*var_28, **var_29)
    var_31 = [var_26, var_30]
    var_32 = [var_31]
    var_33 = {}
    var_34 = module_0.Set(*var_32, **var_33)
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'set[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = 3
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Constant(*var_10, **var_11)
    var_13 = 4
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Constant(*var_14, **var_15)
    var_17 = [var_12, var_16]
    var_18 = [var_8, var_17]
    var_19 = {}
    var_20 = module_0.Dict(*var_18, **var_19)
    var_21 = module_1.const_type(var_20)
    assert var_21 == 'dict[int, int]'
    var_22 = [var_0]
    var_23 = {}
    var_24 = module_0.Constant(*var_22, **var_23)
    var_25 = [var_4]
    var_26 = {}
    var_27 = module_0.Constant(*var_25, **var_26)
    var_28 = [var_24, var_27]
    var_29 = [var_9]
    var_30 = {}
    var_31 = module_0.Constant(*var_29, **var_30)
    var_32 = [var_13]
    var_33 = {}
    var_34 = module_0.Constant(*var_32, **var_33)
    var_35 = [var_31, var_34]
    var_36 = [var_28, var_35]
    var_37 = {}
    var_38 = module_0.Dict(*var_36, **var_37)
    var_39 = module_1.const_type(var_38)
    assert var_39 == 'dict[Any, Any]'
    var_40 = [var_0]
    var_41 = {}
    var_42 = module_0.Constant(*var_40, **var_41)
    var_43 = 'hello'
    var_44 = [var_43]
    var_45 = {}
    var_46 = module_0.Constant(*var_44, **var_45)
    var_47 = [var_42, var_46]
    var_48 = [var_9]
    var_49 = {}
    var_50 = module_0.Constant(*var_48, **var_49)
    var_51 = [var_13]
    var_52 = {}
    var_53 = module_0.Constant(*var_51, **var_52)
    var_54 = [var_50, var_53]
    var_55 = [var_47, var_54]
    var_56 = {}
    var_57 = module_0.Dict(*var_55, **var_56)
    var_58 = module_1.const_type(var_57)
    assert var_58 == 'dict[Any, Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'bool'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Name(*var_1, **var_2)
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.Call(*var_5, **var_6)
    var_8 = module_1.const_type(var_7)
    assert var_8 == 'bool'
    var_9 = 'int'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Name(*var_10, **var_11)
    var_13 = []
    var_14 = [var_12, var_13]
    var_15 = {}
    var_16 = module_0.Call(*var_14, **var_15)
    var_17 = module_1.const_type(var_16)
    assert var_17 == 'int'
    var_18 = 'float'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_0.Name(*var_19, **var_20)
    var_22 = []
    var_23 = [var_21, var_22]
    var_24 = {}
    var_25 = module_0.Call(*var_23, **var_24)
    var_26 = module_1.const_type(var_25)
    assert var_26 == 'float'
    var_27 = 'complex'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_0.Name(*var_28, **var_29)
    var_31 = []
    var_32 = [var_30, var_31]
    var_33 = {}
    var_34 = module_0.Call(*var_32, **var_33)
    var_35 = module_1.const_type(var_34)
    assert var_35 == 'complex'
    var_36 = 'str'
    var_37 = [var_36]
    var_38 = {}
    var_39 = module_0.Name(*var_37, **var_38)
    var_40 = []
    var_41 = [var_39, var_40]
    var_42 = {}
    var_43 = module_0.Call(*var_41, **var_42)
    var_44 = module_1.const_type(var_43)
    assert var_44 == 'str'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Name(*var_1, **var_2)
    var_4 = module_1.const_type(var_3)
    assert var_4 == 'Any'
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Name(*var_5, **var_6)
    var_8 = 'y'
    var_9 = [var_7, var_8]
    var_10 = {}
    var_11 = module_0.Attribute(*var_9, **var_10)
    var_12 = module_1.const_type(var_11)
    assert var_12 == 'Any'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_attr_simple_attribute. Retrieved 1/6 statements.
# Partially parsed test_attr_nested_attribute. Retrieved 1/10 statements.
# Partially parsed test_attr_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_nonexistent_nested_attribute. Retrieved 1/9 statements.
# Partially parsed test_attr_middle_nonexistent_attribute. Retrieved 1/5 statements.
# Partially parsed test_attr_empty_string. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'simple_attr'

def test_case_0():
    var_0 = 'inner.nested_attr'

def test_case_0():
    var_0 = 'nonexistent_attr'

def test_case_0():
    var_0 = 'inner.nonexistent_attr'

def test_case_0():
    var_0 = 'nonexistent_attr.nested_attr'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_api_function. Retrieved 10/13 statements.
# Partially parsed test_api_async_function. Retrieved 10/13 statements.
# Partially parsed test_api_with_decorators. Retrieved 13/16 statements.
# Partially parsed test_api_with_docstring. Retrieved 14/17 statements.
# Partially parsed test_api_with_prefix. Retrieved 11/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'test_module.test_func'
    var_12 = bool('test_module.test_func' in var_0.doc)
    assert var_12 is True
    var_13 = '## test_func()'
    var_14 = bool('## test_func()' in var_0.doc['test_module.test_func'])
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_async_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'test_module.test_async_func'
    var_12 = bool('test_module.test_async_func' in var_0.doc)
    assert var_12 is True
    var_13 = '## async test_async_func()'
    var_14 = bool('## async test_async_func()' in var_0.doc['test_module.test_async_func'])
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TestClass'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = 'name'
    var_8 = 'bases'
    var_9 = 'body'
    var_10 = 'decorator_list'
    var_11 = {var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_1.ClassDef(*var_6, **var_11)
    var_13 = var_0.api(var_1, var_12)
    var_14 = 'test_module.TestClass'
    var_15 = bool('test_module.TestClass' in var_0.doc)
    assert var_15 is True
    var_16 = '## class TestClass'
    var_17 = bool('## class TestClass' in var_0.doc['test_module.TestClass'])
    assert var_17 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'decorator'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = 'test_func'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = [var_10]
    var_20 = 'test_module.test_func'
    var_21 = bool('test_module.test_func' in var_0.doc)
    assert var_21 is True
    var_22 = 'Decorators'
    var_23 = bool('Decorators' in var_0.doc['test_module.test_func'])
    assert var_23 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'Test docstring'
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_15}
    var_19 = module_1.Expr(*var_16, **var_18)
    var_20 = [var_19]
    var_21 = 'test_module.test_func'
    var_22 = bool('test_module.test_func' in var_0.doc)
    assert var_22 is True
    var_23 = 'Test docstring'
    var_24 = bool('Test docstring' in var_0.docstring['test_module.test_func'])
    assert var_24 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = None
    var_10 = []
    var_11 = 'ClassName'
    var_12 = 'test_module.ClassName.test_func'
    var_13 = bool('test_module.ClassName.test_func' in var_0.doc)
    assert var_13 is True
    var_14 = '### test_func()'
    var_15 = bool('### test_func()' in var_0.doc['test_module.ClassName.test_func'])
    assert var_15 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 12/13 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'x'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_1.Name(*var_3, **var_5)
    var_7 = [var_6]
    var_8 = 1
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = 'int'
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = 'type_comment'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.Assign(*var_14, **var_18)
    var_20 = var_0.globals(var_1, var_19)
    var_21 = 'root.x'
    var_22 = 'ANY'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__defaults_with_ampersand_in_expression. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' '])
    assert var_4 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 42
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_0.Constant(*var_6, **var_8)
    var_10 = [var_4, var_9]
    var_11 = module_1._defaults(var_10)
    var_12 = list(var_11)
    var_13 = bool(var_12 == ['`x`', '`42`'])
    assert var_13 is True

import ast as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'b'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = {}
    var_7 = module_0.BitOr(*var_5, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_0.Name(*var_9, **var_11)
    var_13 = []
    var_14 = 'left'
    var_15 = 'op'
    var_16 = 'right'
    var_17 = {var_14: var_4, var_15: var_7, var_16: var_12}
    var_18 = module_0.BinOp(*var_13, **var_17)
    var_19 = [var_18]
    var_20 = module_1._defaults(var_19)
    var_21 = list(var_20)
    var_22 = bool(var_21 == ['`x &#124; y`'])
    assert var_22 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = None
    var_1 = 'var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_0.Name(*var_2, **var_4)
    var_6 = 100
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_0.Constant(*var_7, **var_9)
    var_11 = [var_0, var_5, var_10, var_0]
    var_12 = module_1._defaults(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [' ', '`var`', '`100`', ' '])
    assert var_14 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_walk_body_with_try_statement_and_handler. Retrieved 7/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = module_1.walk_body(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [var_2])
    assert var_6 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.If(*var_2, **var_3)
    var_5 = [var_4]
    var_6 = module_1.walk_body(var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.If(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = module_1.walk_body(var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [var_2])
    assert var_11 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = []
    var_4 = [var_2]
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.If(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = module_1.walk_body(var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [var_2])
    assert var_11 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_0.stmt(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = [var_5]
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_0.If(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = module_1.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [var_2, var_5])
    assert var_14 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = {}
    var_7 = module_0.Try(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = module_1.walk_body(var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.Try(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = module_1.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [var_2])
    assert var_14 is True

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = []
    var_4 = [var_2]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = []
    var_4 = []
    var_5 = [var_2]
    var_6 = []
    var_7 = []
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.Try(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = module_1.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [var_2])
    assert var_14 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = [var_2]
    var_7 = []
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.Try(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = module_1.walk_body(var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [var_2])
    assert var_14 is True

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.If(*var_5, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_0.stmt(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = {}
    var_18 = module_0.Try(*var_16, **var_17)
    var_19 = [var_7, var_18]
    var_20 = module_1.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = bool(var_21 == [var_2, var_10])
    assert var_22 is True



# Parsed testcases at query #21
#--------------------------




import builtins as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'nonexistent_attribute'
    var_4 = module_1._attr(var_2, var_3)
    assert var_4 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_public_returns_false_when_all_l_is_empty. Retrieved 5/7 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = 'test'
    var_3 = set()
    var_4 = var_0.is_public(var_1)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_isinstance_node_Try. Retrieved 5/6 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'body'
    var_6 = 'handlers'
    var_7 = 'orelse'
    var_8 = 'finalbody'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Try(*var_4, **var_9)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 8/12 statements.


import ast as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = []
    var_7 = 'int'
    var_8 = 'float'
    var_9 = 'complex'
    var_10 = 'str'
    var_11 = {var_0, var_7, var_8, var_9, var_10}



# Parsed testcases at query #25
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'def foo():\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = '# Module `test_module`'
    var_7 = bool('# Module `test_module`' in var_0.doc['test_module'])
    assert var_7 is True
    var_8 = 'test_module'
    var_9 = bool('test_module' in var_0.level)
    assert var_9 is True
    var_10 = var_0.level['test_module']
    assert var_10 == 0
    var_11 = 'test_module'
    var_12 = bool('test_module' in var_0.root)
    assert var_12 is True
    var_13 = var_0.root['test_module']
    assert var_13 == 'test_module'
    var_14 = 'test_module'
    var_15 = bool('test_module' in var_0.imp)
    assert var_15 is True
    var_16 = var_0.imp[var_1]
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = 'test_module.foo'
    var_19 = bool('test_module.foo' in var_0.doc)
    assert var_19 is True
    var_20 = '# test_module.foo()'
    var_21 = bool('# test_module.foo()' in var_0.doc['test_module.foo'])
    assert var_21 is True
    var_22 = 'test_module.foo'
    var_23 = bool('test_module.foo' in var_0.level)
    assert var_23 is True
    var_24 = var_0.level['test_module.foo']
    assert var_24 == 0
    var_25 = 'test_module.foo'
    var_26 = bool('test_module.foo' in var_0.root)
    assert var_26 is True
    var_27 = var_0.root['test_module.foo']
    assert var_27 == 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'import os\nfrom sys import path\nx = 1'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.alias)
    assert var_5 is True
    var_6 = var_0.alias['test_module.os']
    assert var_6 == 'os'
    var_7 = var_0.alias['test_module.path']
    assert var_7 == 'sys.path'
    var_8 = 'test_module.x'
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    var_10 = var_0.alias['test_module.x']
    assert var_10 == '1'
    var_11 = 'test_module.x'
    var_12 = bool('test_module.x' in var_0.const)
    assert var_12 is True
    var_13 = var_0.const['test_module.x']
    assert var_13 == 'int'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = 'Module docstring.'
    var_7 = bool('Module docstring.' in var_0.docstring['test_module'])
    assert var_7 is True
    var_8 = 'test_module.foo'
    var_9 = bool('test_module.foo' in var_0.docstring)
    assert var_9 is True
    var_10 = 'Function docstring.'
    var_11 = bool('Function docstring.' in var_0.docstring['test_module.foo'])
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(var_0)
    var_2 = 'test_module'
    var_3 = 'def foo():\n    pass'
    var_4 = var_1.parse(var_2, var_3)
    var_5 = '<a id="test_module"></a>'
    var_6 = bool('<a id="test_module"></a>' in var_1.doc['test_module'])
    assert var_6 is True
    var_7 = '<a id="test_module-foo"></a>'
    var_8 = bool('<a id="test_module-foo"></a>' in var_1.doc['test_module.foo'])
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    var_2 = var_1.link
    assert var_2 is True
    var_3 = 'test_module'
    var_4 = 'def foo():\n    pass'
    var_5 = var_1.parse(var_3, var_4)
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_1.doc)
    assert var_7 is True
    var_8 = 'test_module.foo'
    var_9 = bool('test_module.foo' in var_1.doc)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_3 = var_0.parse(var_1, var_2)
    var_4 = 'test_module.Foo'
    var_5 = bool('test_module.Foo' in var_0.doc)
    assert var_5 is True
    var_6 = 'class test_module.Foo'
    var_7 = bool('class test_module.Foo' in var_0.doc['test_module.Foo'])
    assert var_7 is True
    var_8 = 'test_module.Foo.bar'
    var_9 = bool('test_module.Foo.bar' in var_0.doc)
    assert var_9 is True
    var_10 = 'test_module.Foo.baz'
    var_11 = bool('test_module.Foo.baz' in var_0.doc)
    assert var_11 is True
    var_12 = 'test_module.Foo'
    var_13 = bool('test_module.Foo' in var_0.root)
    assert var_13 is True
    var_14 = var_0.root['test_module.Foo']
    assert var_14 == 'test_module'
    var_15 = 'test_module.Foo.bar'
    var_16 = bool('test_module.Foo.bar' in var_0.root)
    assert var_16 is True
    var_17 = var_0.root['test_module.Foo.bar']
    assert var_17 == 'test_module'
    var_18 = 'test_module.Foo.baz'
    var_19 = bool('test_module.Foo.baz' in var_0.root)
    assert var_19 is True
    var_20 = var_0.root['test_module.Foo.baz']
    assert var_20 == 'test_module'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test__attr_returns_none_for_nonexistent_nested_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent.attr'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_globals_with_ann_assign. Retrieved 9/13 statements.
# Partially parsed test_globals_with_assign. Retrieved 6/11 statements.
# Partially parsed test_globals_with_assign_no_type_comment. Retrieved 5/10 statements.
# Partially parsed test_globals_with_all. Retrieved 8/13 statements.
# Partially parsed test_globals_with_non_uppercase. Retrieved 5/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 'int'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 42
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    var_17 = 1
    var_18 = var_0.alias['test_module.TEST_CONST']
    assert var_18 == '42'
    var_19 = var_0.const['test_module.TEST_CONST']
    assert var_19 == 'int'
    var_20 = var_0.root['test_module.TEST_CONST']
    assert var_20 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 42
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 'int'
    var_9 = var_0.alias['test_module.TEST_CONST']
    assert var_9 == '42'
    var_10 = var_0.const['test_module.TEST_CONST']
    assert var_10 == 'int'
    var_11 = var_0.root['test_module.TEST_CONST']
    assert var_11 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'TEST_CONST'
    var_3 = 42
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_0.alias['test_module.TEST_CONST']
    assert var_8 == '42'
    var_9 = var_0.const['test_module.TEST_CONST']
    assert var_9 == 'int'
    var_10 = var_0.root['test_module.TEST_CONST']
    assert var_10 == 'test_module'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '__all__'
    var_3 = 'public_func'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'elts'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.List(*var_12, **var_15)
    var_17 = var_0.imp['test_module']
    var_18 = bool(var_0.imp['test_module'] == {'test_module.public_func'})
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'non_upper'
    var_3 = 42
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = var_0.alias['test_module.non_upper']
    assert var_8 == '42'
    var_9 = 'test_module.non_upper'
    var_10 = bool('test_module.non_upper' not in var_0.const)
    assert var_10 is True
    var_11 = 'test_module.non_upper'
    var_12 = bool('test_module.non_upper' not in var_0.root)
    assert var_12 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 19/20 statements.
# Partially parsed test_class_api_with_enum. Retrieved 16/17 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'x: int'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = 'y = 1'
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = 'del z'
    var_15 = module_1.parse(var_14, mode=var_4)
    var_16 = var_15.body
    var_17 = [var_10, var_13, var_16]
    var_18 = var_0.class_api(var_1, var_2, var_7, var_17)
    var_19 = 'Bases'
    var_20 = bool('Bases' in var_0.doc[var_2])
    assert var_20 is True
    var_21 = 'Members'
    var_22 = bool('Members' in var_0.doc[var_2])
    assert var_22 is True
    var_23 = 'x'
    var_24 = bool('x' in var_0.doc[var_2])
    assert var_24 is True
    var_25 = 'y'
    var_26 = bool('y' in var_0.doc[var_2])
    assert var_26 is True
    var_27 = 'z'
    var_28 = bool('z' not in var_0.doc[var_2])
    assert var_28 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'A = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = 'B = 2'
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = [var_10, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_7, var_14)
    var_16 = 'Enums'
    var_17 = bool('Enums' in var_0.doc[var_2])
    assert var_17 is True
    var_18 = 'A'
    var_19 = bool('A' in var_0.doc[var_2])
    assert var_19 is True
    var_20 = 'B'
    var_21 = bool('B' in var_0.doc[var_2])
    assert var_21 is True
    var_22 = 'Members'
    var_23 = bool('Members' not in var_0.doc[var_2])
    assert var_23 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_class_api_deletes_enum_attribute. Retrieved 20/45 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'TestClass'
    var_2 = 'enum'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_10]
    var_12 = 'VALUE1'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'id'
    var_19 = 'ctx'
    var_20 = {var_18: var_13, var_19: var_16}
    var_21 = module_1.Name(*var_17, **var_20)
    var_22 = 1
    var_23 = []
    var_24 = 'value'
    var_25 = {var_24: var_22}
    var_26 = module_1.Constant(*var_23, **var_25)
    var_27 = 'VALUE2'
    var_28 = []
    var_29 = {}
    var_30 = module_1.Load(*var_28, **var_29)
    var_31 = []
    var_32 = 'id'
    var_33 = 'ctx'
    var_34 = {var_32: var_13, var_33: var_30}
    var_35 = module_1.Name(*var_31, **var_34)
    var_36 = 2
    var_37 = []
    var_38 = 'value'
    var_39 = {var_38: var_36}
    var_40 = module_1.Constant(*var_37, **var_39)
    var_41 = []
    var_42 = 'test_module'
    var_43 = 'test_module.TestClass'
    var_44 = 'VALUE1'
    var_45 = bool('VALUE1' not in var_0.doc['test_module.TestClass'])
    assert var_45 is True
    var_46 = 'VALUE2'
    var_47 = bool('VALUE2' in var_0.doc['test_module.TestClass'])
    assert var_47 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_is_public_with_all_l. Retrieved 6/8 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'root'
    var_2 = 'child'
    var_3 = 'parent.child'
    var_4 = {var_2, var_3}
    var_5 = var_0.is_public(var_3)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_imports_with_level. Retrieved 6/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'path'
    var_3 = None
    var_4 = 1
    var_5 = 'pkg.subpkg'
    var_6 = var_0.alias['pkg.subpkg.path']
    assert var_6 == 'os.path'



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_compile_empty. Retrieved 2/4 statements.
# Partially parsed test_compile_with_toc. Retrieved 1/8 statements.
# Partially parsed test_compile_without_toc. Retrieved 2/9 statements.
# Partially parsed test_compile_with_magic_method. Retrieved 2/9 statements.
# Partially parsed test_compile_with_non_public. Retrieved 2/9 statements.
# Partially parsed test_compile_with_constants. Retrieved 2/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 1

def test_case_0():
    var_0 = True
    var_1 = '**Table of contents:**'
    var_2 = '+ [root](#root)'
    var_3 = 'Root docstring'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = '**Table of contents:**'
    var_3 = 'Root docstring'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'Init docstring'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'Private docstring'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'Constants'
    var_3 = 'CONST'
    var_4 = 'int'



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = [var_1, var_2, var_3, var_4, var_2, var_5, var_2]
    var_7 = {}
    var_8 = module_1.arguments(*var_6, **var_7)
    var_9 = 'root'
    var_10 = 'test_func'
    var_11 = False
    var_12 = var_0.func_api(var_9, var_10, var_8, var_2, has_self=var_11, cls_method=var_11)
    var_13 = var_0.doc['root.test_func']
    var_14 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |')
    assert var_14 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'y'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = [var_10, var_2, var_11, var_12, var_2, var_13, var_2]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = 'root'
    var_18 = 'test_func'
    var_19 = False
    var_20 = var_0.func_api(var_17, var_18, var_16, var_2, has_self=var_19, cls_method=var_19)
    var_21 = var_0.doc['root.test_func']
    var_22 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|x|y|return|\n|:---:|:---:|:---:|\n| | | |')
    assert var_22 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'y'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = 'z'
    var_12 = [var_11, var_2]
    var_13 = {}
    var_14 = module_1.arg(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = 1
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Constant(*var_17, **var_18)
    var_20 = [var_19]
    var_21 = []
    var_22 = [var_10, var_2, var_15, var_20, var_2, var_21, var_2]
    var_23 = {}
    var_24 = module_1.arguments(*var_22, **var_23)
    var_25 = 'root'
    var_26 = 'test_func'
    var_27 = False
    var_28 = var_0.func_api(var_25, var_26, var_24, var_2, has_self=var_27, cls_method=var_27)
    var_29 = var_0.doc['root.test_func']
    var_30 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|x|y|z|return|\n|:---:|:---:|:---:|:---:|\n| | |`1`| |\n| | | | |')
    assert var_30 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = 'args'
    var_6 = [var_5, var_2]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = []
    var_10 = [var_1, var_2, var_3, var_4, var_8, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'test_func'
    var_15 = False
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = var_0.doc['root.test_func']
    var_18 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|*args|return|\n|:---:|:---:|\n| | |')
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = 'x'
    var_6 = [var_5, var_2]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'y'
    var_10 = [var_9, var_2]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = [var_1, var_2, var_3, var_4, var_2, var_13, var_2]
    var_15 = {}
    var_16 = module_1.arguments(*var_14, **var_15)
    var_17 = 'root'
    var_18 = 'test_func'
    var_19 = False
    var_20 = var_0.func_api(var_17, var_18, var_16, var_2, has_self=var_19, cls_method=var_19)
    var_21 = var_0.doc['root.test_func']
    var_22 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|*|x|y|return|\n|:---:|:---:|:---:|:---:|\n| | | | |')
    assert var_22 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'kwargs'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_1, var_2, var_3, var_4, var_2, var_5, var_9]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'test_func'
    var_15 = False
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = var_0.doc['root.test_func']
    var_18 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|**kwargs|return|\n|:---:|:---:|\n| | |')
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = None
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = [var_1, var_2, var_3, var_4, var_2, var_5, var_2]
    var_7 = {}
    var_8 = module_1.arguments(*var_6, **var_7)
    var_9 = 'root'
    var_10 = 'test_func'
    var_11 = 'int'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Constant(*var_12, **var_13)
    var_15 = False
    var_16 = var_0.func_api(var_9, var_10, var_8, var_14, has_self=var_15, cls_method=var_15)
    var_17 = var_0.doc['root.test_func']
    var_18 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n|`int`|')
    assert var_18 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_2, var_7, var_8, var_2, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'test_func'
    var_15 = True
    var_16 = False
    var_17 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_16)
    var_18 = var_0.doc['root.test_func']
    var_19 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |')
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_2, var_7, var_8, var_2, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    var_13 = 'root'
    var_14 = 'test_func'
    var_15 = True
    var_16 = var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)
    var_17 = var_0.doc['root.test_func']
    var_18 = bool(var_0.doc['root.test_func'] == '#' * (var_0.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |')
    assert var_18 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_imports_without_asname. Retrieved 5/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = 'path'
    var_3 = 0
    var_4 = 'pkg'
    var_5 = var_0.alias['pkg.path']
    assert var_5 == 'sys.path'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_docstring_basic. Retrieved 6/13 statements.
# Partially parsed test_load_docstring_nested. Retrieved 12/19 statements.
# Partially parsed test_load_docstring_none_doc. Retrieved 5/11 statements.
# Partially parsed test_load_docstring_no_match. Retrieved 5/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.func'
    var_2 = 'pkg.submod'
    var_3 = 'Function doc'
    var_4 = 'Module doc'
    var_5 = None
    var_6 = var_0.docstring['pkg.submod.func']
    assert var_6 == 'New function doc'
    var_7 = 'pkg.submod'
    var_8 = bool('pkg.submod' not in var_0.docstring)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.Class.method'
    var_2 = 'pkg.submod.Class'
    var_3 = 'Method doc'
    var_4 = 'Class doc'
    var_5 = 'pkg.submod'
    var_6 = 'Class'
    var_7 = ()
    var_8 = 'method'
    var_9 = None
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = var_0.docstring['pkg.submod.Class.method']
    assert var_12 == 'New method doc'
    var_13 = 'pkg.submod.Class'
    var_14 = bool('pkg.submod.Class' not in var_0.docstring)
    assert var_14 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.func'
    var_2 = 'Function doc'
    var_3 = 'pkg.submod'
    var_4 = None
    var_5 = 'pkg.submod.func'
    var_6 = bool('pkg.submod.func' not in var_0.docstring)
    assert var_6 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.submod.func'
    var_2 = 'Function doc'
    var_3 = 'pkg.submod'
    var_4 = None
    var_5 = 'pkg.submod.func'
    var_6 = bool('pkg.submod.func' not in var_0.docstring)
    assert var_6 is True
    var_7 = 'pkg.submod.other_func'
    var_8 = bool('pkg.submod.other_func' not in var_0.docstring)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_1, var_10]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    var_14 = 'x'
    var_15 = 'int'
    var_16 = []
    var_17 = {}
    var_18 = module_1.Load(*var_16, **var_17)
    var_19 = []
    var_20 = 'id'
    var_21 = 'ctx'
    var_22 = {var_20: var_15, var_21: var_18}
    var_23 = module_1.Name(*var_19, **var_22)
    var_24 = [var_14, var_23]
    var_25 = {}
    var_26 = module_1.arg(*var_24, **var_25)
    var_27 = [var_13, var_26]
    var_28 = 'module'
    var_29 = True
    var_30 = False
    var_31 = var_0.func_ann(var_28, var_27, has_self=var_29, cls_method=var_30)
    var_32 = list(var_31)
    var_33 = bool(var_32 == ['Self', 'int'])
    assert var_33 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_1, var_10]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    var_14 = 'x'
    var_15 = 'int'
    var_16 = []
    var_17 = {}
    var_18 = module_1.Load(*var_16, **var_17)
    var_19 = []
    var_20 = 'id'
    var_21 = 'ctx'
    var_22 = {var_20: var_15, var_21: var_18}
    var_23 = module_1.Name(*var_19, **var_22)
    var_24 = [var_14, var_23]
    var_25 = {}
    var_26 = module_1.arg(*var_24, **var_25)
    var_27 = [var_13, var_26]
    var_28 = 'module'
    var_29 = True
    var_30 = var_0.func_ann(var_28, var_27, has_self=var_29, cls_method=var_29)
    var_31 = list(var_30)
    var_32 = bool(var_31 == ['type[Self]', 'int'])
    assert var_32 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'y'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = 'module'
    var_12 = False
    var_13 = var_0.func_ann(var_11, var_10, has_self=var_12, cls_method=var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['ANY', 'ANY'])
    assert var_15 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_1, var_10]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    var_14 = '*'
    var_15 = None
    var_16 = [var_14, var_15]
    var_17 = {}
    var_18 = module_1.arg(*var_16, **var_17)
    var_19 = 'y'
    var_20 = 'str'
    var_21 = []
    var_22 = {}
    var_23 = module_1.Load(*var_21, **var_22)
    var_24 = []
    var_25 = 'id'
    var_26 = 'ctx'
    var_27 = {var_25: var_20, var_26: var_23}
    var_28 = module_1.Name(*var_24, **var_27)
    var_29 = [var_19, var_28]
    var_30 = {}
    var_31 = module_1.arg(*var_29, **var_30)
    var_32 = [var_13, var_18, var_31]
    var_33 = 'module'
    var_34 = False
    var_35 = var_0.func_ann(var_33, var_32, has_self=var_34, cls_method=var_34)
    var_36 = list(var_35)
    var_37 = bool(var_36 == ['int', '', 'str'])
    assert var_37 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__attr_with_single_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_with_nested_attributes. Retrieved 2/7 statements.
# Partially parsed test__attr_with_nonexistent_attribute. Retrieved 2/5 statements.
# Partially parsed test__attr_with_partial_nonexistent_nested_attribute. Retrieved 2/7 statements.
# Partially parsed test__attr_with_deeply_nested_attributes. Retrieved 2/9 statements.
# Partially parsed test__attr_with_empty_attribute_string. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'attr1'

def test_case_0():
    var_0 = 'value2'
    var_1 = 'attr1.attr2'

def test_case_0():
    var_0 = 'value1'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'value2'
    var_1 = 'attr1.nonexistent'

def test_case_0():
    var_0 = 'value3'
    var_1 = 'attr1.attr2.attr3'

def test_case_0():
    var_0 = 'value1'
    var_1 = ''



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_func_api_with_kwonlyargs. Retrieved 16/18 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = 'kw_arg'
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = None
    var_15 = False
    var_16 = False
    var_17 = '*'
    var_18 = [var_17, var_5]
    var_19 = {}
    var_20 = module_1.arg(*var_18, **var_19)
    var_21 = bool(var_20 in var_0.doc[var_2])
    assert var_21 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_class_api_with_enum_base. Retrieved 16/30 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.EnumClass'
    var_2 = 'test'
    var_3 = 1
    var_4 = '# test.EnumClass\n\n'
    var_5 = set()
    var_6 = 'EnumClass'
    var_7 = 'Enum'
    var_8 = []
    var_9 = {}
    var_10 = module_1.Load(*var_8, **var_9)
    var_11 = []
    var_12 = 'id'
    var_13 = 'ctx'
    var_14 = {var_12: var_7, var_13: var_10}
    var_15 = module_1.Name(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = 'RED'
    var_18 = 'int'
    var_19 = []
    var_20 = {}
    var_21 = module_1.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_18, var_24: var_21}
    var_26 = module_1.Name(*var_22, **var_25)
    var_27 = []
    var_28 = 'value'
    var_29 = {var_28: var_3}
    var_30 = module_1.Constant(*var_27, **var_29)
    var_31 = 'Enums'
    var_32 = bool('Enums' in var_0.doc['test.EnumClass'])
    assert var_32 is True
    var_33 = 'RED'
    var_34 = bool('RED' in var_0.doc['test.EnumClass'])
    assert var_34 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_api_no_recursion_on_non_class. Retrieved 12/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_func'
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = var_0.doc
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'test_module.test_func'
    var_14 = bool('test_module.test_func' in var_0.doc)
    assert var_14 is True



# Parsed testcases at query #44
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'root'
    var_8 = True
    var_9 = False
    var_10 = var_0.func_ann(var_7, var_6, has_self=var_8, cls_method=var_9)
    var_11 = list(var_10)
    var_12 = var_11[0]
    assert var_12 == 'Self'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_false. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_globals_const_not_any. Retrieved 10/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'CONST'
    var_3 = 'int'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    var_8 = 42
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    var_13 = 1
    var_14 = 'test_module.CONST'
    var_15 = 'ANY'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 12/14 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 12/14 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 11/13 statements.
# Partially parsed test_func_api_with_kwargs. Retrieved 8/10 statements.
# Partially parsed test_func_api_with_defaults. Retrieved 13/15 statements.
# Partially parsed test_func_api_with_self. Retrieved 10/12 statements.
# Partially parsed test_func_api_with_cls_method. Retrieved 9/11 statements.
# Partially parsed test_func_api_with_annotations. Retrieved 15/17 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 9/11 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'b'
    var_8 = [var_7, var_2]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'root'
    var_15 = 'name'
    var_16 = False
    var_17 = var_0.doc['name']
    assert var_17 == '| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = 'b'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = [var_6, var_10]
    var_12 = []
    var_13 = []
    var_14 = 'root'
    var_15 = 'name'
    var_16 = False
    var_17 = var_0.doc['name']
    assert var_17 == '| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'args'
    var_8 = [var_7, var_2]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'name'
    var_15 = False
    var_16 = var_0.doc['name']
    assert var_16 == '| a | *args | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'kwargs'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = []
    var_8 = 'root'
    var_9 = 'name'
    var_10 = False
    var_11 = var_0.doc['name']
    assert var_11 == '| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'b'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = 1
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Constant(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = []
    var_17 = 'root'
    var_18 = 'name'
    var_19 = False
    var_20 = var_0.doc['name']
    assert var_20 == '| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `1` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = True
    var_12 = False
    var_13 = var_0.doc['name']
    assert var_13 == '| self | return |\n|:---:|:---:|\n| `Self` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'cls'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = 'root'
    var_10 = 'name'
    var_11 = True
    var_12 = var_0.doc['name']
    assert var_12 == '| cls | return |\n|:---:|:---:|\n| `type[Self]` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = 'int'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.Name(*var_4, **var_5)
    var_7 = [var_1, var_6]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = 'b'
    var_11 = 'str'
    var_12 = [var_11, var_3]
    var_13 = {}
    var_14 = module_1.Name(*var_12, **var_13)
    var_15 = [var_10, var_14]
    var_16 = {}
    var_17 = module_1.arg(*var_15, **var_16)
    var_18 = [var_9, var_17]
    var_19 = []
    var_20 = []
    var_21 = 'root'
    var_22 = 'name'
    var_23 = False
    var_24 = var_0.doc['name']
    assert var_24 == '| a | b | return |\n|:---:|:---:|:---:|\n| `int` | `str` | `Any` |\n\n'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'root'
    var_5 = 'name'
    var_6 = 'bool'
    var_7 = None
    var_8 = [var_6, var_7]
    var_9 = {}
    var_10 = module_1.Name(*var_8, **var_9)
    var_11 = False
    var_12 = var_0.doc['name']
    assert var_12 == '| return |\n|:---:|\n| `bool` |\n\n'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_func_api_with_kwarg. Retrieved 14/16 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_function'
    var_3 = []
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []
    var_8 = 'kwargs'
    var_9 = [var_8, var_5]
    var_10 = {}
    var_11 = module_1.arg(*var_9, **var_10)
    var_12 = []
    var_13 = []
    var_14 = None
    var_15 = False
    var_16 = False
    var_17 = '**kwargs'
    var_18 = bool('**kwargs' in var_0.doc[var_2])
    assert var_18 is True



# Parsed testcases at query #49
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''
    var_4 = [var_0, var_0]
    var_5 = [var_4]
    var_6 = module_0._e_type(*var_5)
    assert var_6 == ''

import apimd.parser as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_1]
    var_3 = module_0._e_type(*var_2)
    assert var_3 == ''
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = [var_5]
    var_7 = module_0._e_type(*var_6)
    assert var_7 == ''

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = module_1._e_type(*var_5)
    assert var_6 == '[int]'
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Constant(*var_7, **var_8)
    var_10 = 2
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Constant(*var_11, **var_12)
    var_14 = [var_9, var_13]
    var_15 = [var_14]
    var_16 = module_1._e_type(*var_15)
    assert var_16 == '[int]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Constant(*var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_1._e_type(*var_9)
    assert var_10 == '[Any]'
    var_11 = [var_0]
    var_12 = {}
    var_13 = module_0.Constant(*var_11, **var_12)
    var_14 = 2
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_0.Constant(*var_15, **var_16)
    var_18 = [var_13, var_17]
    var_19 = [var_18]
    var_20 = module_1._e_type(*var_19)
    assert var_20 == '[Any]'

import ast as module_0
import apimd.parser as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Constant(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 2
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Constant(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = [var_4, var_9]
    var_11 = module_1._e_type(*var_10)
    assert var_11 == '[int, int]'
    var_12 = [var_0]
    var_13 = {}
    var_14 = module_0.Constant(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = 'a'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_0.Constant(*var_17, **var_18)
    var_20 = [var_19]
    var_21 = [var_15, var_20]
    var_22 = module_1._e_type(*var_21)
    assert var_22 == '[int, str]'
    var_23 = [var_0]
    var_24 = {}
    var_25 = module_0.Constant(*var_23, **var_24)
    var_26 = [var_25]
    var_27 = [var_0]
    var_28 = {}
    var_29 = module_0.Constant(*var_27, **var_28)
    var_30 = [var_29]
    var_31 = [var_26, var_30]
    var_32 = module_1._e_type(*var_31)
    assert var_32 == '[int, Any]'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_none_attribute_access. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'alias'
    var_3 = 'root'
    var_4 = 'root.alias'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_walk_body_with_try_node. Retrieved 8/15 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.stmt(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.stmt(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = {}
    var_10 = module_0.stmt(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = {}
    var_14 = module_0.stmt(*var_12, **var_13)
    var_15 = [var_14]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_imports_with_asname. Retrieved 5/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = 'operating_system'
    var_3 = 'pkg'
    var_4 = 'pkg.operating_system'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_toc_true. Retrieved 3/4 statements.


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
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_visit_Name_with_self_ty. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_alias. Retrieved 9/10 statements.
# Partially parsed test_visit_Name_without_alias. Retrieved 7/8 statements.
# Partially parsed test_visit_Name_with_TypeVar_alias. Retrieved 9/10 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = 'MyClass'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = var_3.visit_Name(var_9)
    var_11 = var_10.id
    assert var_11 == 'Self'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.MyType'
    var_2 = 'int'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'int'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    var_3 = 'MyType'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    var_10 = var_2.visit_Name(var_9)
    var_11 = var_10.id
    assert var_11 == 'MyType'

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.MyType'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    var_5 = 'MyType'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    var_12 = var_4.visit_Name(var_11)
    var_13 = var_12.id
    assert var_13 == 'MyType'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_public_with_all_l. Retrieved 8/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = 'subpkg.submod'
    var_4 = {var_2, var_3}
    var_5 = 'pkg.subpkg'
    var_6 = 'pkg.subpkg.submod'
    var_7 = var_0.is_public(var_5)
    assert var_7 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_class_api. Retrieved 16/29 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = 'attr1'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    var_23 = None
    var_24 = 'attr2'
    var_25 = 42
    var_26 = []
    var_27 = 'value'
    var_28 = {var_27: var_25}
    var_29 = module_1.Constant(*var_26, **var_28)
    var_30 = 'attr3'
    var_31 = 'Bases'
    var_32 = bool('Bases' in var_0.doc[var_2])
    assert var_32 is True
    var_33 = 'attr1'
    var_34 = bool('attr1' in var_0.doc[var_2])
    assert var_34 is True
    var_35 = 'attr2'
    var_36 = bool('attr2' in var_0.doc[var_2])
    assert var_36 is True
    var_37 = 'attr3'
    var_38 = bool('attr3' not in var_0.doc[var_2])
    assert var_38 is True



# Parsed testcases at query #59
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'x: int = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = "y: str = 'hello'"
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = 'z = [1, 2, 3]'
    var_15 = module_1.parse(var_14, mode=var_4)
    var_16 = var_15.body
    var_17 = [var_10, var_13, var_16]
    var_18 = var_0.class_api(var_1, var_2, var_7, var_17)
    var_19 = '# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id="test-module-testclass"></a>\n\n'
    var_20 = 'Bases'
    var_21 = [var_3]
    var_22 = [var_20]
    var_23 = module_0.table(*var_22, items=var_21)
    var_24 = var_19 + var_23
    var_25 = 'Members'
    var_26 = 'Type'
    var_27 = 'x'
    var_28 = 'int'
    var_29 = (var_27, var_28)
    var_30 = 'y'
    var_31 = 'str'
    var_32 = (var_30, var_31)
    var_33 = 'z'
    var_34 = 'list[int]'
    var_35 = (var_33, var_34)
    var_36 = [var_29, var_32, var_35]
    var_37 = [var_25, var_26]
    var_38 = module_0.table(*var_37, items=var_36)
    var_39 = var_24 + var_38
    var_40 = var_0.doc[var_2]
    var_41 = bool(var_0.doc[var_2] == var_39)
    assert var_41 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 'enum.Enum'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = 'A = 1'
    var_9 = module_1.parse(var_8, mode=var_4)
    var_10 = var_9.body
    var_11 = 'B = 2'
    var_12 = module_1.parse(var_11, mode=var_4)
    var_13 = var_12.body
    var_14 = [var_10, var_13]
    var_15 = var_0.class_api(var_1, var_2, var_7, var_14)
    var_16 = '# class TestEnum\n\n*Full name:* `test_module.TestEnum`\n<a id="test-module-testenum"></a>\n\n'
    var_17 = 'Enums'
    var_18 = 'A'
    var_19 = 'B'
    var_20 = [var_18, var_19]
    var_21 = [var_17]
    var_22 = module_0.table(*var_21, items=var_20)
    var_23 = var_16 + var_22
    var_24 = var_0.doc[var_2]
    var_25 = bool(var_0.doc[var_2] == var_23)
    assert var_25 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 'x: int = 1'
    var_5 = 'eval'
    var_6 = module_1.parse(var_4, mode=var_5)
    var_7 = var_6.body
    var_8 = 'del x'
    var_9 = module_1.parse(var_8, mode=var_5)
    var_10 = var_9.body
    var_11 = [var_7, var_10]
    var_12 = var_0.class_api(var_1, var_2, var_3, var_11)
    var_13 = '# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id="test-module-testclass"></a>\n\n'
    var_14 = var_0.doc[var_2]
    var_15 = bool(var_0.doc[var_2] == var_13)
    assert var_15 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_load_docstring. Retrieved 6/13 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg'
    var_2 = 'pkg.func'
    var_3 = '# Module `pkg`'
    var_4 = '## func()'
    var_5 = None
    var_6 = var_0.docstring['pkg']
    assert var_6 == 'Package doc'
    var_7 = var_0.docstring['pkg.func']
    assert var_7 == 'Function doc'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_is_magic_predicate. Retrieved 5/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__init__'
    var_2 = '# Module `__init__`\n\n'
    var_3 = 0
    var_4 = var_0.compile()
    var_5 = '__init__'
    var_6 = bool('__init__' not in var_4)
    assert var_6 is True



# Parsed testcases at query #62
#--------------------------




import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    var_6 = [var_5]
    var_7 = 'foo'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = 'elts'
    var_15 = {var_14: var_12}
    var_16 = module_1.Tuple(*var_13, **var_15)
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    var_22 = 'root'
    var_23 = var_0.globals(var_22, var_21)
    var_24 = '__all__'
    var_25 = bool('__all__' not in var_0.imp['root'])
    assert var_25 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_func_api_with_positional_args. Retrieved 19/22 statements.
# Partially parsed test_func_api_with_keyword_args. Retrieved 17/20 statements.
# Partially parsed test_func_api_with_varargs. Retrieved 16/19 statements.
# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 16/19 statements.
# Partially parsed test_func_api_with_return_annotation. Retrieved 14/17 statements.


import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.arg(*var_4, **var_5)
    var_7 = 'b'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_0.arg(*var_8, **var_9)
    var_11 = [var_6, var_10]
    var_12 = 'c'
    var_13 = [var_12, var_3]
    var_14 = {}
    var_15 = module_0.arg(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = [var_1]
    var_18 = {}
    var_19 = module_0.Constant(*var_17, **var_18)
    var_20 = 2
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_0.Constant(*var_21, **var_22)
    var_24 = [var_19, var_23]
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = 'root'
    var_29 = 'root.func'
    var_30 = '| a | b | / | c | return |\n|:---:|:---:|:---:|:---:|:---:|\n| `int` | `int` |  |  |  |\n| 1 | 2 |  |  |  |'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'a'
    var_6 = None
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_0.arg(*var_7, **var_8)
    var_10 = 'b'
    var_11 = [var_10, var_6]
    var_12 = {}
    var_13 = module_0.arg(*var_11, **var_12)
    var_14 = [var_9, var_13]
    var_15 = [var_1]
    var_16 = {}
    var_17 = module_0.Constant(*var_15, **var_16)
    var_18 = 2
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_0.Constant(*var_19, **var_20)
    var_22 = [var_17, var_21]
    var_23 = []
    var_24 = 'root'
    var_25 = 'root.func'
    var_26 = '| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `int` | `int` |  |\n|  | 1 | 2 |  |'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'a'
    var_4 = None
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.arg(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'kwargs'
    var_13 = [var_12, var_4]
    var_14 = {}
    var_15 = module_0.arg(*var_13, **var_14)
    var_16 = 'args'
    var_17 = [var_16, var_4]
    var_18 = {}
    var_19 = module_0.arg(*var_17, **var_18)
    var_20 = []
    var_21 = 'root'
    var_22 = 'root.func'
    var_23 = '| a | *args | **kwargs | return |\n|:---:|:---:|:---:|:---:|\n|  |  |  |  |'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'self'
    var_4 = None
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.arg(*var_5, **var_6)
    var_8 = 'a'
    var_9 = [var_8, var_4]
    var_10 = {}
    var_11 = module_0.arg(*var_9, **var_10)
    var_12 = [var_7, var_11]
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = 'root'
    var_18 = 'root.func'
    var_19 = True
    var_20 = True
    var_21 = '| type[Self] | a | return |\n|:---:|:---:|:---:|\n|  |  |  |'

import ast as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = []
    var_3 = 'a'
    var_4 = None
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.arg(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'root'
    var_14 = 'root.func'
    var_15 = 'str'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Constant(*var_16, **var_17)
    var_19 = '| a | return |\n|:---:|:---:|\n|  | `str` |'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_func_api_with_vararg. Retrieved 17/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module'
    var_2 = 'module.func'
    var_3 = []
    var_4 = 'x'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.arg(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = 'args'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = None
    var_18 = False
    var_19 = False
    var_20 = var_0.doc[var_2]
    var_21 = '| Arguments | Type |\n| --- | --- |\n| x | Any |\n| *args | Any |\n| return | Any |\n'



# Parsed testcases at query #66
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



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_load_docstring_updates_docstring. Retrieved 13/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.sub'
    var_2 = 'pkg.other'
    var_3 = '...'
    var_4 = 'Module'
    var_5 = ()
    var_6 = 'sub'
    var_7 = 'Sub module doc'
    var_8 = {var_6: var_7}
    var_9 = type(var_4, var_5, var_8)
    var_10 = var_9()
    var_11 = 'pkg'
    var_12 = var_0.load_docstring(var_11, var_10)
    var_13 = 'pkg.sub'
    var_14 = bool('pkg.sub' in var_0.docstring)
    assert var_14 is True
    var_15 = var_0.docstring['pkg.sub']
    assert var_15 == 'Sub module doc'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test__attr_returns_none_for_missing_nested_attribute. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'non.existent.attribute'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_class_api_with_bases_and_members. Retrieved 16/26 statements.
# Partially parsed test_class_api_with_enum. Retrieved 23/32 statements.
# Partially parsed test_class_api_with_deleted_members. Retrieved 10/20 statements.
# Partially parsed test_class_api_with_private_members. Retrieved 13/23 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.ClassName'
    var_2 = 'BaseClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_10]
    var_12 = 'public_attr'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'id'
    var_19 = 'ctx'
    var_20 = {var_18: var_13, var_19: var_16}
    var_21 = module_1.Name(*var_17, **var_20)
    var_22 = None
    var_23 = 1
    var_24 = 'another_attr'
    var_25 = 42
    var_26 = []
    var_27 = 'value'
    var_28 = {var_27: var_25}
    var_29 = module_1.Constant(*var_26, **var_28)
    var_30 = 'test.module'
    var_31 = 'Bases'
    var_32 = bool('Bases' in var_0.doc[var_1])
    assert var_32 is True
    var_33 = 'Members'
    var_34 = bool('Members' in var_0.doc[var_1])
    assert var_34 is True
    var_35 = 'public_attr'
    var_36 = bool('public_attr' in var_0.doc[var_1])
    assert var_36 is True
    var_37 = 'another_attr'
    var_38 = bool('another_attr' in var_0.doc[var_1])
    assert var_38 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.EnumClass'
    var_2 = 'enum.Enum'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    var_11 = [var_10]
    var_12 = 'FIRST'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_1.Load(*var_14, **var_15)
    var_17 = []
    var_18 = 'id'
    var_19 = 'ctx'
    var_20 = {var_18: var_13, var_19: var_16}
    var_21 = module_1.Name(*var_17, **var_20)
    var_22 = 1
    var_23 = []
    var_24 = 'value'
    var_25 = {var_24: var_22}
    var_26 = module_1.Constant(*var_23, **var_25)
    var_27 = []
    var_28 = {}
    var_29 = module_1.Load(*var_27, **var_28)
    var_30 = []
    var_31 = 'value'
    var_32 = 'slice'
    var_33 = 'ctx'
    var_34 = {var_31: var_21, var_32: var_26, var_33: var_29}
    var_35 = module_1.Subscript(*var_30, **var_34)
    var_36 = None
    var_37 = 'SECOND'
    var_38 = []
    var_39 = {}
    var_40 = module_1.Load(*var_38, **var_39)
    var_41 = []
    var_42 = 'id'
    var_43 = 'ctx'
    var_44 = {var_42: var_13, var_43: var_40}
    var_45 = module_1.Name(*var_41, **var_44)
    var_46 = 2
    var_47 = []
    var_48 = 'value'
    var_49 = {var_48: var_46}
    var_50 = module_1.Constant(*var_47, **var_49)
    var_51 = []
    var_52 = {}
    var_53 = module_1.Load(*var_51, **var_52)
    var_54 = []
    var_55 = 'value'
    var_56 = 'slice'
    var_57 = 'ctx'
    var_58 = {var_55: var_45, var_56: var_50, var_57: var_53}
    var_59 = module_1.Subscript(*var_54, **var_58)
    var_60 = 'test.module'
    var_61 = 'Enums'
    var_62 = bool('Enums' in var_0.doc[var_1])
    assert var_62 is True
    var_63 = 'FIRST'
    var_64 = bool('FIRST' in var_0.doc[var_1])
    assert var_64 is True
    var_65 = 'SECOND'
    var_66 = bool('SECOND' in var_0.doc[var_1])
    assert var_66 is True
    var_67 = 'Members'
    var_68 = bool('Members' not in var_0.doc[var_1])
    assert var_68 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.ClassName'
    var_2 = []
    var_3 = 'public_attr'
    var_4 = 'int'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = None
    var_14 = 1
    var_15 = 'test.module'
    var_16 = 'Members'
    var_17 = bool('Members' not in var_0.doc[var_1])
    assert var_17 is True
    var_18 = 'public_attr'
    var_19 = bool('public_attr' not in var_0.doc[var_1])
    assert var_19 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module.ClassName'
    var_2 = []
    var_3 = '_private_attr'
    var_4 = 'int'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    var_13 = None
    var_14 = 1
    var_15 = 'public_attr'
    var_16 = 42
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = 'test.module'
    var_22 = '_private_attr'
    var_23 = bool('_private_attr' not in var_0.doc[var_1])
    assert var_23 is True
    var_24 = 'public_attr'
    var_25 = bool('public_attr' in var_0.doc[var_1])
    assert var_25 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_func_api_with_kwonlyargs. Retrieved 18/20 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = None
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.arg(*var_6, **var_7)
    var_9 = 'b'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_8, var_12]
    var_14 = 1
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Constant(*var_15, **var_16)
    var_18 = 2
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Constant(*var_19, **var_20)
    var_22 = [var_17, var_21]
    var_23 = []
    var_24 = 'root'
    var_25 = 'name'
    var_26 = False
    var_27 = '*'
    var_28 = bool('*' in var_0.doc['root.name'])
    assert var_28 is True



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_parser_new_method. Retrieved 3/4 statements.
# Partially parsed test_parser_post_init_with_toc. Retrieved 3/4 statements.


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

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_func_api_with_posonlyargs_and_defaults. Retrieved 18/20 statements.
# Partially parsed test_func_api_with_vararg_and_kwarg. Retrieved 18/20 statements.
# Partially parsed test_func_api_with_self_and_cls_method. Retrieved 14/16 statements.
# Partially parsed test_func_api_with_returns. Retrieved 13/15 statements.


import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'a'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    var_6 = 'b'
    var_7 = [var_6, var_2]
    var_8 = {}
    var_9 = module_1.arg(*var_7, **var_8)
    var_10 = [var_5, var_9]
    var_11 = 'c'
    var_12 = [var_11, var_2]
    var_13 = {}
    var_14 = module_1.arg(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = 1
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = []
    var_25 = 'root'
    var_26 = 'func'
    var_27 = False
    var_28 = '| a | b | / | c | return |'
    var_29 = bool('| a | b | / | c | return |' in var_0.doc['root.func'])
    assert var_29 is True
    var_30 = '|:---:|:---:|:---:|:---:|:---:|'
    var_31 = bool('|:---:|:---:|:---:|:---:|:---:|' in var_0.doc['root.func'])
    assert var_31 is True
    var_32 = '| `a` | `b` |  | `c` | `Any` |'
    var_33 = bool('| `a` | `b` |  | `c` | `Any` |' in var_0.doc['root.func'])
    assert var_33 is True
    var_34 = '|  |  |  | `1` |  |'
    var_35 = bool('|  |  |  | `1` |  |' in var_0.doc['root.func'])
    assert var_35 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'a'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = 'b'
    var_10 = [var_9, var_3]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'kwargs'
    var_16 = [var_15, var_3]
    var_17 = {}
    var_18 = module_1.arg(*var_16, **var_17)
    var_19 = 'args'
    var_20 = [var_19, var_3]
    var_21 = {}
    var_22 = module_1.arg(*var_20, **var_21)
    var_23 = []
    var_24 = 'root'
    var_25 = 'func'
    var_26 = False
    var_27 = '| a | * | b | **kwargs | return |'
    var_28 = bool('| a | * | b | **kwargs | return |' in var_0.doc['root.func'])
    assert var_28 is True
    var_29 = '|:---:|:---:|:---:|:---:|:---:|'
    var_30 = bool('|:---:|:---:|:---:|:---:|:---:|' in var_0.doc['root.func'])
    assert var_30 is True
    var_31 = '| `a` |  | `b` |  | `Any` |'
    var_32 = bool('| `a` |  | `b` |  | `Any` |' in var_0.doc['root.func'])
    assert var_32 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = 'self'
    var_3 = None
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_1.arg(*var_4, **var_5)
    var_7 = 'a'
    var_8 = [var_7, var_3]
    var_9 = {}
    var_10 = module_1.arg(*var_8, **var_9)
    var_11 = [var_6, var_10]
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'root'
    var_17 = 'func'
    var_18 = True
    var_19 = '| self | a | return |'
    var_20 = bool('| self | a | return |' in var_0.doc['root.func'])
    assert var_20 is True
    var_21 = '|:---:|:---:|:---:|'
    var_22 = bool('|:---:|:---:|:---:|' in var_0.doc['root.func'])
    assert var_22 is True
    var_23 = '| `type[Self]` | `Any` | `Any` |'
    var_24 = bool('| `type[Self]` | `Any` | `Any` |' in var_0.doc['root.func'])
    assert var_24 is True

import apimd.parser as module_0
import ast as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = None
    var_7 = []
    var_8 = 'int'
    var_9 = []
    var_10 = {}
    var_11 = module_1.Load(*var_9, **var_10)
    var_12 = []
    var_13 = 'id'
    var_14 = 'ctx'
    var_15 = {var_13: var_8, var_14: var_11}
    var_16 = module_1.Name(*var_12, **var_15)
    var_17 = 'root'
    var_18 = 'func'
    var_19 = False
    var_20 = '| return |'
    var_21 = bool('| return |' in var_0.doc['root.func'])
    assert var_21 is True
    var_22 = '|:---:|'
    var_23 = bool('|:---:|' in var_0.doc['root.func'])
    assert var_23 is True
    var_24 = '| `int` |'
    var_25 = bool('| `int` |' in var_0.doc['root.func'])
    assert var_25 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 8/11 statements.


import ast as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = []
    var_6 = 'func'
    var_7 = {var_6: var_4}
    var_8 = module_0.Call(*var_5, **var_7)
    var_9 = 'int'
    var_10 = 'float'
    var_11 = 'complex'
    var_12 = 'str'
    var_13 = {var_0, var_9, var_10, var_11, var_12}



