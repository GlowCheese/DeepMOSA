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



