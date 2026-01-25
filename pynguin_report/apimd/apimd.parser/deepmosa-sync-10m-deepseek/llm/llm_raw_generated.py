####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test__m_with_empty_args():
    assert _m() == ""

def test__m_with_single_arg():
    assert _m("module") == "module"

def test__m_with_multiple_args():
    assert _m("module", "submodule", "function") == "module.submodule.function"

def test__m_with_empty_strings():
    assert _m("module", "", "function") == "module.function"

def test__m_with_all_empty_strings():
    assert _m("", "", "") == ""


# LLM-generated content at query #2
#--------------------------

```python
def test_walk_body_with_single_stmt():
    stmt = "stmt"
    result = list(walk_body([stmt]))
    assert result == [stmt]

def test_walk_body_with_if_stmt():
    if_stmt = If(body=["stmt1"], orelse=["stmt2"])
    result = list(walk_body([if_stmt]))
    assert result == ["stmt1", "stmt2"]

def test_walk_body_with_try_stmt():
    try_stmt = Try(body=["stmt1"], handlers=[excepthandler(body=["stmt2"])], orelse=["stmt3"], finalbody=["stmt4"])
    result = list(walk_body([try_stmt]))
    assert result == ["stmt1", "stmt2", "stmt3", "stmt4"]

def test_walk_body_with_nested_if_and_try():
    if_stmt = If(body=[Try(body=["stmt1"], handlers=[excepthandler(body=["stmt2"])], orelse=["stmt3"], finalbody=["stmt4"])], orelse=["stmt5"])
    result = list(walk_body([if_stmt]))
    assert result == ["stmt1", "stmt2", "stmt3", "stmt4", "stmt5"]

def test_walk_body_with_multiple_stmts():
    stmt1 = "stmt1"
    stmt2 = "stmt2"
    stmt3 = "stmt3"
    result = list(walk_body([stmt1, stmt2, stmt3]))
    assert result == [stmt1, stmt2, stmt3]


# LLM-generated content at query #3
#--------------------------

```
def test_globals_handles_ann_assign_with_value():
    p = Parser()
    node = AnnAssign(target=Name(id='x'), annotation=Name(id='int'), value=Constant(value=42))
    p.globals('root', node)
    assert p.alias['root.x'] == '42'
    assert p.const['root.x'] == 'int'

def test_globals_handles_assign_with_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='y')], value=Constant(value='test'), type_comment='str')
    p.globals('root', node)
    assert p.alias['root.y'] == "'test'"
    assert p.const['root.y'] == 'str'

def test_globals_handles_assign_without_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z')], value=Constant(value=3.14))
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_handles_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='PI')], value=Constant(value=3.14159))
    p.globals('root', node)
    assert p.alias['root.PI'] == '3.14159'
    assert p.const['root.PI'] == 'float'

def test_globals_handles_all_special_case():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=List(elts=[Constant(value='x'), Constant(value='y')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.x', 'root.y'}

def test_globals_ignores_non_name_assign():
    p = Parser()
    node = Assign(targets=[Subscript()], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const

def test_globals_ignores_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a'), Name(id='b')], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const


# LLM-generated content at query #4
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link == False
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link == True

def test_parser_constructor_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link == True


# LLM-generated content at query #5
#--------------------------

```python
def test_imports_method_import():
    parser = Parser()
    parser.parse("root", "")
    node = Import(names=[ast.alias(name="module", asname=None)])
    parser.imports("root", node)
    assert parser.alias == {"root.module": "module"}

def test_imports_method_import_with_alias():
    parser = Parser()
    parser.parse("root", "")
    node = Import(names=[ast.alias(name="module", asname="alias")])
    parser.imports("root", node)
    assert parser.alias == {"root.alias": "module"}

def test_imports_method_import_from():
    parser = Parser()
    parser.parse("root", "")
    node = ImportFrom(module="module", names=[ast.alias(name="name", asname=None)], level=0)
    parser.imports("root", node)
    assert parser.alias == {"root.name": "module.name"}

def test_imports_method_import_from_with_alias():
    parser = Parser()
    parser.parse("root", "")
    node = ImportFrom(module="module", names=[ast.alias(name="name", asname="alias")], level=0)
    parser.imports("root", node)
    assert parser.alias == {"root.alias": "module.name"}

def test_imports_method_import_from_with_level():
    parser = Parser()
    parser.parse("root.sub", "")
    node = ImportFrom(module="module", names=[ast.alias(name="name", asname=None)], level=1)
    parser.imports("root.sub", node)
    assert parser.alias == {"root.sub.name": "root.module.name"}

def test_imports_method_import_from_with_level_and_alias():
    parser = Parser()
    parser.parse("root.sub", "")
    node = ImportFrom(module="module", names=[ast.alias(name="name", asname="alias")], level=1)
    parser.imports("root.sub", node)
    assert parser.alias == {"root.sub.alias": "root.module.name"}


# LLM-generated content at query #6
#--------------------------

```python
def test__defaults_with_none():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_string():
    args = ["test", "example", "data"]
    result = list(_defaults(args))
    assert result == ["`test`", "`example`", "`data`"]

def test__defaults_with_string_containing_ampersand():
    args = ["test&", "example&", "data&"]
    result = list(_defaults(args))
    assert result == ["<code>test&#38;</code>", "<code>example&#38;</code>", "<code>data&#38;</code>"]

def test__defaults_with_string_containing_pipe():
    args = ["test|", "example|", "data|"]
    result = list(_defaults(args))
    assert result == ["`test&#124;`", "`example&#124;`", "`data&#124;`"]

def test__defaults_with_mixed_types():
    args = [None, "test", "example&", "data|"]
    result = list(_defaults(args))
    assert result == [" ", "`test`", "<code>example&#38;</code>", "`data&#124;`"]


# LLM-generated content at query #7
#--------------------------

```
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    p.globals('root', node)
    assert p.alias['root.x'] == '42'
    assert p.const['root.x'] == 'int'

def test_globals_with_assign():
    p = Parser()
    node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value='test'))
    p.globals('root', node)
    assert p.alias['root.y'] == "'test'"
    assert p.const['root.y'] == 'str'

def test_globals_with_assign_and_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14), type_comment='float')
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_with_non_name_target():
    p = Parser()
    node = Assign(targets=[Subscript(value=Name(id='arr', ctx=Load()), slice=Constant(value=0))], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.arr' not in p.alias
    assert 'root.arr' not in p.const

def test_globals_with_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.a' not in p.alias
    assert 'root.b' not in p.alias

def test_globals_with_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=100))
    p.globals('root', node)
    assert p.alias['root.CONST'] == '100'
    assert p.const['root.CONST'] == 'int'

def test_globals_with_non_constant_value():
    p = Parser()
    node = Assign(targets=[Name(id='func', ctx=Store())], value=Call(func=Name(id='len', ctx=Load()), args=[Constant(value='test')]))
    p.globals('root', node)
    assert p.alias['root.func'] == 'len(\'test\')'
    assert p.const['root.func'] == 'Any'

def test_globals_with_all_assignment():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='func'), Constant(value='CONST')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.func', 'root.CONST'}


# LLM-generated content at query #8
#--------------------------

```
def test_parse_simple_module():
    p = Parser()
    p.parse("test_module", "def foo(): pass")
    assert "test_module" in p.doc
    assert "test_module" in p.level
    assert "test_module" in p.root
    assert "test_module" in p.imp
    assert p.doc["test_module"].startswith("# Module `test_module`")

def test_parse_with_imports():
    p = Parser()
    p.parse("test_module", "import os\ndef foo(): pass")
    assert "os" in p.alias
    assert "_m(test_module, os)" in p.alias

def test_parse_with_import_from():
    p = Parser()
    p.parse("test_module", "from os import path\ndef foo(): pass")
    assert "path" in p.alias
    assert "_m(test_module, path)" in p.alias

def test_parse_with_function_def():
    p = Parser()
    p.parse("test_module", "def foo(): pass")
    assert "_m(test_module, foo)" in p.doc
    assert p.doc["_m(test_module, foo)"].startswith("### foo()")

def test_parse_with_class_def():
    p = Parser()
    p.parse("test_module", "class Foo: pass")
    assert "_m(test_module, Foo)" in p.doc
    assert p.doc["_m(test_module, Foo)"].startswith("### class Foo")

def test_parse_with_global_vars():
    p = Parser()
    p.parse("test_module", "CONST = 42\n__all__ = ['foo']\ndef foo(): pass")
    assert "_m(test_module, CONST)" in p.const
    assert "_m(test_module, foo)" in p.imp["test_module"]

def test_parse_with_docstring():
    p = Parser()
    p.parse("test_module", "\"\"\"Module docstring\"\"\"\ndef foo(): pass")
    assert "test_module" in p.docstring
    assert "Module docstring" in p.docstring["test_module"]

def test_parse_with_nested_classes():
    p = Parser()
    code = "class Outer:\n    class Inner: pass"
    p.parse("test_module", code)
    assert "_m(test_module, Outer)" in p.doc
    assert "_m(test_module, Outer.Inner)" in p.doc


# LLM-generated content at query #9
#--------------------------

```python
def test_empty_elements():
    assert _e_type() == ""

def test_none_element():
    assert _e_type(None) == ""

def test_non_constant_element():
    class NonConstant:
        pass
    assert _e_type([NonConstant()]) == ""

def test_single_constant_element():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)]) == "[int]"

def test_multiple_constant_elements_same_type():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test_multiple_constant_elements_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test_multiple_elements_with_none():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), None, Constant(2)]) == ""

def test_multiple_sequences_of_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)], [Constant(3), Constant(4)]) == "[int, int]"

def test_multiple_sequences_of_constants_mixed_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)], [Constant("a"), Constant("b")]) == "[int, str]"


# LLM-generated content at query #10
#--------------------------

```python
def test_imports_with_node_module_none():
    parser = Parser()
    node = Import(names=[alias(name='test_module', asname=None)])
    parser.imports('root', node)
    assert node.module is None


# LLM-generated content at query #11
#--------------------------

```python
def test_globals_assign_with_type_comment():
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=42), type_comment='int')
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.x'] == 'int'

def test_globals_ann_assign_with_annotation():
    node = AnnAssign(target=Name(id='y', ctx=Store()), annotation=Name(id='str', ctx=Load()), value=Constant(value='hello'))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.y'] == 'str'

def test_globals_assign_without_type_comment():
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.z'] == 'float'

def test_globals_assign_with_complex_value():
    node = Assign(targets=[Name(id='w', ctx=Store())], value=Call(func=Name(id='complex', ctx=Load()), args=[Constant(value=1), Constant(value=2)], keywords=[]))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.w'] == 'complex'

def test_globals_assign_with_list_value():
    node = Assign(targets=[Name(id='lst', ctx=Store())], value=List(elts=[Constant(value=1), Constant(value=2)], ctx=Load()))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.lst'] == 'list[int, int]'

def test_globals_assign_with_dict_value():
    node = Assign(targets=[Name(id='dct', ctx=Store())], value=Dict(keys=[Constant(value='a')], values=[Constant(value=1)]))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.dct'] == 'dict[str, int]'

def test_globals_assign_with_tuple_value():
    node = Assign(targets=[Name(id='tpl', ctx=Store())], value=Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load()))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.tpl'] == 'tuple[int, int]'

def test_globals_assign_with_set_value():
    node = Assign(targets=[Name(id='st', ctx=Store())], value=Set(elts=[Constant(value=1), Constant(value=2)]))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.st'] == 'set[int, int]'

def test_globals_assign_with_non_constant_value():
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Name(id='y', ctx=Load()))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const.get('root.x', 'ANY') == 'ANY'

def test_globals_assign_with_non_name_target():
    node = Assign(targets=[Attribute(value=Name(id='obj', ctx=Load()), attr='x', ctx=Store())], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const.get('root.obj.x', 'ANY') == 'ANY'

def test_globals_assign_with_multiple_targets():
    node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const.get('root.a', 'ANY') == 'ANY'
    assert parser.const.get('root.b', 'ANY') == 'ANY'

def test_globals_assign_with_non_upper_case_name():
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const.get('root.x', 'ANY') == 'ANY'

def test_globals_assign_with_upper_case_name():
    node = Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=42))
    parser = Parser()
    parser.globals('root', node)
    assert parser.const['root.X'] == 'int'

def test_globals_assign_with_all():
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='x'), Constant(value='y')], ctx=Load()))
    parser = Parser()
    parser.globals('root', node)
    assert parser.imp['root'] == {'root.x', 'root.y'}


# LLM-generated content at query #12
#--------------------------

```
def test_globals_ann_assign_with_value():
    p = Parser()
    node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    p.globals('root', node)
    assert p.alias['root.x'] == '42'
    assert p.const['root.x'] == 'int'

def test_globals_assign_with_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value='test'), type_comment='str')
    p.globals('root', node)
    assert p.alias['root.y'] == "'test'"
    assert p.const['root.y'] == 'str'

def test_globals_assign_without_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14))
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_assign_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.a' not in p.alias
    assert 'root.b' not in p.alias

def test_globals_non_name_target():
    p = Parser()
    node = Assign(targets=[Subscript(value=Name(id='lst', ctx=Load()), slice=Constant(value=0), ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.lst' not in p.alias

def test_globals_non_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='var', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.var' in p.alias
    assert 'root.var' not in p.const

def test_globals_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.CONST' in p.alias
    assert 'root.CONST' in p.const

def test_globals_all_special_case():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='x'), Constant(value='y')], ctx=Load()))
    p.globals('root', node)
    assert 'root.x' in p.imp['root']
    assert 'root.y' in p.imp['root']

def test_globals_all_non_list_tuple():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert not p.imp['root']


# LLM-generated content at query #13
#--------------------------

```python
def test__attr_with_single_attribute():
    class TestClass:
        attr = "value"
    obj = TestClass()
    assert _attr(obj, "attr") == "value"

def test__attr_with_nested_attribute():
    class Inner:
        inner_attr = "inner_value"
    class Outer:
        outer_attr = Inner()
    obj = Outer()
    assert _attr(obj, "outer_attr.inner_attr") == "inner_value"

def test__attr_with_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "nonexistent") is None

def test__attr_with_nonexistent_nested_attribute():
    class Inner:
        pass
    class Outer:
        outer_attr = Inner()
    obj = Outer()
    assert _attr(obj, "outer_attr.nonexistent") is None

def test__attr_with_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "") is None

def test__attr_with_none_object():
    assert _attr(None, "attr") is None


# LLM-generated content at query #14
#--------------------------

```
def test_is_public_family_with_public_name():
    assert is_public_family("public.name") == True

def test_is_public_family_with_private_name():
    assert is_public_family("_private.name") == False

def test_is_public_family_with_magic_name():
    assert is_public_family("__magic__") == True

def test_is_public_family_with_mixed_name():
    assert is_public_family("public._private") == False

def test_is_public_family_with_nested_magic_name():
    assert is_public_family("module.__magic__.name") == True

def test_is_public_family_with_empty_string():
    assert is_public_family("") == True


# LLM-generated content at query #15
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.resolve = lambda root, node, self_ty="": "type"
    parser.class_api('root', 'root.Class', [Constant(value='Base')], [AnnAssign(target=Name(id='attr'), annotation=Name(id='int'))])
    assert 'root.Class' in parser.doc
    assert 'Bases' in parser.doc['root.Class']
    assert 'Members' in parser.doc['root.Class']
    assert 'attr' in parser.doc['root.Class']
    assert 'int' in parser.doc['root.Class']

def test_class_api_with_enums():
    parser = Parser()
    parser.doc = {}
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.resolve = lambda root, node, self_ty="": "type"
    parser.class_api('root', 'root.Class', [Constant(value='enum.Enum')], [AnnAssign(target=Name(id='ENUM_VALUE'), annotation=Name(id='int'))])
    assert 'root.Class' in parser.doc
    assert 'Enums' in parser.doc['root.Class']
    assert 'ENUM_VALUE' in parser.doc['root.Class']

def test_class_api_without_bases_and_members():
    parser = Parser()
    parser.doc = {}
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.resolve = lambda root, node, self_ty="": "type"
    parser.class_api('root', 'root.Class', [], [])
    assert 'root.Class' in parser.doc
    assert 'Bases' not in parser.doc['root.Class']
    assert 'Members' not in parser.doc['root.Class']
    assert 'Enums' not in parser.doc['root.Class']

def test_class_api_with_delete_statement():
    parser = Parser()
    parser.doc = {}
    parser.level = {'root': 0}
    parser.root = {'root': 'root'}
    parser.resolve = lambda root, node, self_ty="": "type"
    parser.class_api('root', 'root.Class', [], [AnnAssign(target=Name(id='attr'), annotation=Name(id='int')), Delete(targets=[Name(id='attr')])])
    assert 'root.Class' in parser.doc
    assert 'Members' not in parser.doc['root.Class']


# LLM-generated content at query #16
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    parser.doc = {'root.Class': ''}
    parser.class_api('root', 'root.Class', [Name(id='Base', ctx=Load())], [])
    assert 'Bases' in parser.doc['root.Class']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.doc = {'root.EnumClass': ''}
    parser.class_api('root', 'root.EnumClass', [Name(id='enum.Enum', ctx=Load())], [
        Assign(targets=[Name(id='ENUM_VALUE', ctx=Store())], value=Constant(value=1))
    ])
    assert 'Enums' in parser.doc['root.EnumClass']

def test_class_api_with_members():
    parser = Parser()
    parser.doc = {'root.ClassWithMembers': ''}
    parser.class_api('root', 'root.ClassWithMembers', [], [
        AnnAssign(target=Name(id='member', ctx=Store()), annotation=Name(id='int', ctx=Load()))
    ])
    assert 'Members' in parser.doc['root.ClassWithMembers']

def test_class_api_with_deleted_member():
    parser = Parser()
    parser.doc = {'root.ClassWithDeletedMember': ''}
    parser.class_api('root', 'root.ClassWithDeletedMember', [], [
        AnnAssign(target=Name(id='member', ctx=Store()), annotation=Name(id='int', ctx=Load())),
        Delete(targets=[Name(id='member', ctx=Del())])
    ])
    assert 'Members' not in parser.doc['root.ClassWithDeletedMember']

def test_class_api_with_private_member():
    parser = Parser()
    parser.doc = {'root.ClassWithPrivateMember': ''}
    parser.class_api('root', 'root.ClassWithPrivateMember', [], [
        AnnAssign(target=Name(id='_private', ctx=Store()), annotation=Name(id='int', ctx=Load()))
    ])
    assert 'Members' not in parser.doc['root.ClassWithPrivateMember']


# LLM-generated content at query #17
#--------------------------

```python
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(
        posonlyargs=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| x | y | / | return |' in p.doc['func_name']
    assert '|:---:|:---:|:---:|:---:|' in p.doc['func_name']

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='a', annotation=None), arg(arg='b', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[None, Constant(value=1)]
    )
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| a | b | return |' in p.doc['func_name']
    assert '|:---:|:---:|:---:|' in p.doc['func_name']
    assert '|  | 1 |  |' in p.doc['func_name']

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg(arg='args', annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| *args | return |' in p.doc['func_name']
    assert '|:---:|:---:|' in p.doc['func_name']

def test_func_api_with_kwonlyargs():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg(arg='x', annotation=None), arg(arg='y', annotation=None)],
        kw_defaults=[None, Constant(value=2)],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| * | x | y | return |' in p.doc['func_name']
    assert '|:---:|:---:|:---:|:---:|' in p.doc['func_name']
    assert '|  |  | 2 |  |' in p.doc['func_name']

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg='kwargs', annotation=None),
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| **kwargs | return |' in p.doc['func_name']
    assert '|:---:|:---:|' in p.doc['func_name']

def test_func_api_with_has_self():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='self', annotation=None), arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=True, cls_method=False)
    assert '| Self | x | return |' in p.doc['func_name']
    assert '|:---:|:---:|:---:|' in p.doc['func_name']

def test_func_api_with_cls_method():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='cls', annotation=None), arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, None, has_self=True, cls_method=True)
    assert '| type[Self] | x | return |' in p.doc['func_name']
    assert '|:---:|:---:|:---:|' in p.doc['func_name']

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='x', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'func_name', args, Name(id='int', ctx=Load()), has_self=False, cls_method=False)
    assert '| x | return |' in p.doc['func_name']
    assert '|:---:|:---:|' in p.doc['func_name']
    assert '|  | int |' in p.doc['func_name']


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_4_evaluates_to_true():
    node = If()
    body = [node]
    result = isinstance(body[0], If)
    assert result == True


# LLM-generated content at query #19
#--------------------------

```
def test_const_type_with_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_constant_float():
    node = Constant(value=3.14)
    assert const_type(node) == "float"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_with_constant_bool():
    node = Constant(value=True)
    assert const_type(node) == "bool"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[], ctx=Load())
    assert const_type(node) == "tuple[]"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_mixed_tuple():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "tuple[Any]"

def test_const_type_with_empty_list():
    node = List(elts=[], ctx=Load())
    assert const_type(node) == "list[]"

def test_const_type_with_list_of_floats():
    node = List(elts=[Constant(value=1.1), Constant(value=2.2)], ctx=Load())
    assert const_type(node) == "list[float, float]"

def test_const_type_with_empty_set():
    node = Set(elts=[])
    assert const_type(node) == "set[]"

def test_const_type_with_set_of_strings():
    node = Set(elts=[Constant(value="x"), Constant(value="y")])
    assert const_type(node) == "set[str, str]"

def test_const_type_with_empty_dict():
    node = Dict(keys=[], values=[])
    assert const_type(node) == "dict[]"

def test_const_type_with_dict_of_int_to_str():
    node = Dict(keys=[Constant(value=1)], values=[Constant(value="a")])
    assert const_type(node) == "dict[int, str]"

def test_const_type_with_builtin_func_call():
    node = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "int"

def test_const_type_with_unknown_node():
    node = Name(id="x", ctx=Load())
    assert const_type(node) == "Any"


# LLM-generated content at query #20
#--------------------------

```python
def test__attr_single_level():
    class TestObj:
        attr = "value"
    
    obj = TestObj()
    assert _attr(obj, "attr") == "value"

def test__attr_nested_level():
    class InnerObj:
        inner_attr = "inner_value"
    
    class TestObj:
        nested = InnerObj()
    
    obj = TestObj()
    assert _attr(obj, "nested.inner_attr") == "inner_value"

def test__attr_non_existent_single_level():
    class TestObj:
        pass
    
    obj = TestObj()
    assert _attr(obj, "nonexistent") is None

def test__attr_non_existent_nested_level():
    class InnerObj:
        pass
    
    class TestObj:
        nested = InnerObj()
    
    obj = TestObj()
    assert _attr(obj, "nested.nonexistent") is None

def test__attr_non_existent_intermediate_level():
    class TestObj:
        pass
    
    obj = TestObj()
    assert _attr(obj, "nonexistent.intermediate") is None


# LLM-generated content at query #21
#--------------------------

```python
def test_visit_Subscript_Union():
    resolver = Resolver("typing", {}, "")
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name) and result.right.id == "str"

def test_visit_Subscript_Optional():
    resolver = Resolver("typing", {}, "")
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant) and result.right.value is None

def test_visit_Subscript_PEP585():
    resolver = Resolver("typing", {}, "")
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name) and result.value.id == "list"
    assert isinstance(result.slice, Name) and result.slice.id == "int"

def test_visit_Subscript_other():
    resolver = Resolver("typing", {}, "")
    node = Subscript(Name("Dict", Load()), Tuple([Name("str", Load()), Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result == node


# LLM-generated content at query #22
#--------------------------

```
def test_api_parsing_function_def():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    p.api(root, node)
    assert p.doc["test_module.test_func"] == "### test_func()\n\n*Full name:* `test_module.test_func`\n\n"

def test_api_parsing_async_function_def():
    p = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    p.api(root, node)
    assert p.doc["test_module.test_async_func"] == "### async test_async_func()\n\n*Full name:* `test_module.test_async_func`\n\n"

def test_api_parsing_class_def():
    p = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    p.api(root, node)
    assert p.doc["test_module.TestClass"] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"

def test_api_parsing_with_decorators():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="decorated_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[Name(id="decorator", ctx=Load())], returns=None)
    p.api(root, node)
    assert "| Decorators |\n|:---:|\n| `@decorator` |" in p.doc["test_module.decorated_func"]

def test_api_parsing_with_prefix():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="method", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    p.api(root, node, prefix="TestClass")
    assert p.doc["test_module.TestClass.method"] == "#### method()\n\n*Full name:* `test_module.TestClass.method`\n\n"

def test_api_parsing_with_docstring():
    p = Parser()
    root = "test_module"
    node = FunctionDef(name="doc_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[Expr(value=Constant(value="This is a docstring"))], decorator_list=[], returns=None)
    p.api(root, node)
    assert "This is a docstring" in p.docstring["test_module.doc_func"]


# LLM-generated content at query #23
#--------------------------

```
def test_e_type_empty_elements():
    assert _e_type() == ""


# LLM-generated content at query #24
#--------------------------

```
def test_imports_with_node_level():
    parser = Parser()
    node = ImportFrom(module='module', names=[alias(name='name')], level=1)
    parser.imports('root', node)
    assert 'root.name' in parser.alias


# LLM-generated content at query #25
#--------------------------

```python
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='x', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| x |' in p.doc['func_name']

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| *args |' in p.doc['func_name']

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=arg(arg='kwargs', annotation=None), defaults=[])
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| **kwargs |' in p.doc['func_name']

def test_func_api_with_has_self():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'func_name', args, None, has_self=True, cls_method=False)
    assert '| Self |' in p.doc['func_name']

def test_func_api_with_cls_method():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='cls', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'func_name', args, None, has_self=True, cls_method=True)
    assert '| type[Self] |' in p.doc['func_name']

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'func_name', args, Name(id='str', ctx=Load()), has_self=False, cls_method=False)
    assert '| str |' in p.doc['func_name']

def test_func_api_with_default_values():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='x', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[Constant(value=1)])
    p.func_api('root', 'func_name', args, None, has_self=False, cls_method=False)
    assert '| 1 |' in p.doc['func_name']


# LLM-generated content at query #26
#--------------------------

```
def test_class_api_with_bases():
    p = Parser()
    p.parse('test_module', 'class A(B, C): pass')
    assert 'test_module.A' in p.doc
    assert 'Bases' in p.doc['test_module.A']
    assert 'B' in p.doc['test_module.A']
    assert 'C' in p.doc['test_module.A']

def test_class_api_with_enums():
    p = Parser()
    p.parse('test_module', 'class A(enum.Enum): X = 1; Y = 2')
    assert 'test_module.A' in p.doc
    assert 'Enums' in p.doc['test_module.A']
    assert 'X' in p.doc['test_module.A']
    assert 'Y' in p.doc['test_module.A']

def test_class_api_with_members():
    p = Parser()
    p.parse('test_module', 'class A: x: int; y: str')
    assert 'test_module.A' in p.doc
    assert 'Members' in p.doc['test_module.A']
    assert 'x' in p.doc['test_module.A']
    assert 'int' in p.doc['test_module.A']
    assert 'y' in p.doc['test_module.A']
    assert 'str' in p.doc['test_module.A']

def test_class_api_with_deleted_members():
    p = Parser()
    p.parse('test_module', 'class A: x: int; del x')
    assert 'test_module.A' in p.doc
    assert 'Members' not in p.doc['test_module.A']

def test_class_api_with_private_members():
    p = Parser()
    p.parse('test_module', 'class A: _x: int; y: str')
    assert 'test_module.A' in p.doc
    assert 'Members' in p.doc['test_module.A']
    assert '_x' not in p.doc['test_module.A']
    assert 'y' in p.doc['test_module.A']


# LLM-generated content at query #27
#--------------------------

```python
def test__e_type_empty_elements():
    assert _e_type() == ""

def test__e_type_empty_sequence():
    assert _e_type([]) == ""

def test__e_type_non_constant_element():
    assert _e_type([object()]) == ""

def test__e_type_single_constant_element():
    assert _e_type([Constant(42)]) == "[int]"

def test__e_type_multiple_constant_elements_same_type():
    assert _e_type([Constant(42), Constant(24)]) == "[int, int]"

def test__e_type_multiple_constant_elements_different_types():
    assert _e_type([Constant(42), Constant("24")]) == "[Any]"

def test__e_type_multiple_sequences_same_type():
    assert _e_type([Constant(42), Constant(24)], [Constant(99)]) == "[int, int], [int]"

def test__e_type_multiple_sequences_different_types():
    assert _e_type([Constant(42), Constant("24")], [Constant(99)]) == "[Any], [int]"

def test__e_type_mixed_sequences():
    assert _e_type([Constant(42), Constant(24)], [Constant("99")]) == "[int, int], [str]"

def test__e_type_none_element():
    assert _e_type([None]) == ""


# LLM-generated content at query #28
#--------------------------

```python
def test_class_api_with_non_annassign_node():
    parser = Parser()
    root = "test_root"
    name = "test_name"
    bases = []
    body = [Assign(targets=[Name(id="x", ctx=Store())], value=Constant(value=42))]
    parser.class_api(root, name, bases, body)
    assert "Enums" not in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    root = "test_root"
    name = "test_name"
    bases = []
    body = [Assign(targets=[Name(id="x"), Name(id="y")], value=Constant(value=10))]
    parser.class_api(root, name, bases, body)


# LLM-generated content at query #30
#--------------------------

```python
def test_visit_Attribute_with_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_without_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(value=Name(id="module", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result == node

def test_visit_Attribute_with_non_name_value():
    resolver = Resolver("root", {})
    node = Attribute(value=Constant(value="typing"), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result == node


# LLM-generated content at query #31
#--------------------------

```python
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(
        posonlyargs=[arg(arg='arg1', annotation=None), arg(arg='arg2', annotation=None)],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'].count('arg1') == 2
    assert p.doc['name'].count('arg2') == 2
    assert p.doc['name'].count('/') == 1

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='arg1', annotation=None), arg(arg='arg2', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[Constant(value=1), Constant(value=2)]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'].count('arg1') == 2
    assert p.doc['name'].count('arg2') == 2
    assert p.doc['name'].count('1') == 1
    assert p.doc['name'].count('2') == 1

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=arg(arg='args', annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'].count('*args') == 1

def test_func_api_with_kwonlyargs():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg(arg='kw1', annotation=None), arg(arg='kw2', annotation=None)],
        kw_defaults=[Constant(value=1), Constant(value=2)],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'].count('kw1') == 2
    assert p.doc['name'].count('kw2') == 2
    assert p.doc['name'].count('1') == 1
    assert p.doc['name'].count('2') == 1

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg(arg='kwargs', annotation=None),
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert p.doc['name'].count('**kwargs') == 1

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, Name(id='int', ctx=Load()), has_self=False, cls_method=False)
    assert p.doc['name'].count('return') == 1
    assert p.doc['name'].count('int') == 1

def test_func_api_with_self_param():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='self', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert p.doc['name'].count('Self') == 1

def test_func_api_with_classmethod():
    p = Parser()
    args = arguments(
        posonlyargs=[],
        args=[arg(arg='cls', annotation=Name(id='Type', ctx=Load()))],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert p.doc['name'].count('type[Self]') == 1


# LLM-generated content at query #32
#--------------------------

def test_posonlyargs_condition_evaluates_to_true():
    node = arguments(posonlyargs=[arg('arg1', None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    parser = Parser()
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert len(parser.doc['name']) > 0


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_23_evaluates_to_false():
    parser = Parser()
    node = Assign(targets=[Name(id='x')], value=Constant(value=42), type_comment="int")
    parser.globals("root", node)
    assert parser.const.get("root.x", "ANY") == "int"


# LLM-generated content at query #34
#--------------------------

```python
def test_constructor_default_values():
    p = Parser()
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_func_api_with_defaults_and_decorators():
    p = Parser()
    node = FunctionDef(
        name='example',
        args=arguments(
            posonlyargs=[arg(arg='a', annotation=None)],
            args=[arg(arg='b', annotation=None)],
            defaults=[None, None],
            kwonlyargs=[],
            kw_defaults=[],
            vararg=None,
            kwarg=None
        ),
        body=[],
        decorator_list=[Name(id='staticmethod')],
        returns=None
    )
    p.func_api('root', 'root.example', node, returns=None, has_self=False, cls_method=False)
    assert p.doc['root.example'] == "### example()\n\n*Full name:* `root.example`\n\n| a | b |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_vararg_and_kwarg():
    p = Parser()
    node = FunctionDef(
        name='example',
        args=arguments(
            posonlyargs=[],
            args=[],
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            vararg=arg(arg='args', annotation=None),
            kwarg=arg(arg='kwargs', annotation=None)
        ),
        body=[],
        decorator_list=[],
        returns=None
    )
    p.func_api('root', 'root.example', node, returns=None, has_self=False, cls_method=False)
    assert p.doc['root.example'] == "### example()\n\n*Full name:* `root.example`\n\n| * | args | ** | kwargs | return |\n|:---:|:---:|:---:|:---:|:---:|\n|  |  |  |  |  |\n\n"

def test_func_api_with_annotations():
    p = Parser()
    node = FunctionDef(
        name='example',
        args=arguments(
            posonlyargs=[],
            args=[arg(arg='a', annotation=Name(id='int'))],
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            vararg=None,
            kwarg=None
        ),
        body=[],
        decorator_list=[],
        returns=Name(id='str')
    )
    p.func_api('root', 'root.example', node, returns=Name(id='str'), has_self=False, cls_method=False)
    assert p.doc['root.example'] == "### example()\n\n*Full name:* `root.example`\n\n| a | return |\n|:---:|:---:|\n| int | str |\n\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_api_method_for_function_with_decorators():
    class MockNode:
        def __init__(self, name, decorator_list):
            self.name = name
            self.decorator_list = decorator_list
            self.body = []

    parser = Parser()
    node = MockNode("test_function", ["@decorator1", "@decorator2"])
    parser.api("root", node)
    assert parser.doc["root.test_function"] == "### test_function()\n\n*Full name:* `root.test_function`\n\n"

def test_api_method_for_async_function():
    class MockNode:
        def __init__(self, name, decorator_list):
            self.name = name
            self.decorator_list = decorator_list
            self.body = []

    parser = Parser()
    node = MockNode("test_function", [])
    node.__class__.__name__ = "AsyncFunctionDef"
    parser.api("root", node)
    assert parser.doc["root.test_function"] == "### async test_function()\n\n*Full name:* `root.test_function`\n\n"

def test_api_method_for_class():
    class MockNode:
        def __init__(self, name, decorator_list):
            self.name = name
            self.decorator_list = decorator_list
            self.body = []

    parser = Parser()
    node = MockNode("TestClass", [])
    node.__class__.__name__ = "ClassDef"
    parser.api("root", node)
    assert parser.doc["root.TestClass"] == "### class TestClass\n\n*Full name:* `root.TestClass`\n\n"

def test_api_method_for_function_with_self_and_classmethod():
    class MockNode:
        def __init__(self, name, decorator_list):
            self.name = name
            self.decorator_list = decorator_list
            self.body = []

    parser = Parser()
    node = MockNode("test_function", ["@classmethod"])
    parser.api("root", node, prefix="TestClass")
    assert parser.doc["root.TestClass.test_function"] == "#### test_function()\n\n*Full name:* `root.TestClass.test_function`\n\n"

def test_api_method_for_function_with_self_and_staticmethod():
    class MockNode:
        def __init__(self, name, decorator_list):
            self.name = name
            self.decorator_list = decorator_list
            self.body = []

    parser = Parser()
    node = MockNode("test_function", ["@staticmethod"])
    parser.api("root", node, prefix="TestClass")
    assert parser.doc["root.TestClass.test_function"] == "#### test_function()\n\n*Full name:* `root.TestClass.test_function`\n\n"


# LLM-generated content at query #3
#--------------------------

```
def test_api_parses_function_def():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "test_func()" in parser.doc["test_module.test_func"]
    assert "*Full name*: `test_module.test_func`" in parser.doc["test_module.test_func"]

def test_api_parses_async_function_def():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "async test_async_func()" in parser.doc["test_module.test_async_func"]
    assert "*Full name*: `test_module.test_async_func`" in parser.doc["test_module.test_async_func"]

def test_api_parses_class_def():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    assert "*Full name*: `test_module.TestClass`" in parser.doc["test_module.TestClass"]

def test_api_parses_class_def_with_prefix():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node, prefix="prefix")
    assert "class prefix.TestClass" in parser.doc["test_module.prefix.TestClass"]
    assert "*Full name*: `test_module.prefix.TestClass`" in parser.doc["test_module.prefix.TestClass"]

def test_api_parses_function_def_with_decorators():
    parser = Parser()
    root = "test_module"
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[decorator], returns=None)
    parser.api(root, node)
    assert "@decorator" in parser.doc["test_module.test_func"]
    assert "Decorators" in parser.doc["test_module.test_func"]

def test_api_parses_class_def_with_bases():
    parser = Parser()
    root = "test_module"
    base = Name(id="BaseClass", ctx=Load())
    node = ClassDef(name="TestClass", bases=[base], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "Bases" in parser.doc["test_module.TestClass"]
    assert "BaseClass" in parser.doc["test_module.TestClass"]

def test_api_parses_class_def_with_docstring():
    parser = Parser()
    root = "test_module"
    docstring = "This is a test class"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[Expr(value=Constant(value=docstring))], decorator_list=[])
    parser.api(root, node)
    assert docstring in parser.docstring["test_module.TestClass"]

def test_api_parses_function_def_with_docstring():
    parser = Parser()
    root = "test_module"
    docstring = "This is a test function"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[Expr(value=Constant(value=docstring))], decorator_list=[], returns=None)
    parser.api(root, node)
    assert docstring in parser.docstring["test_module.test_func"]


# LLM-generated content at query #4
#--------------------------

```python
def test_doctest_with_single_line():
    doc = ">>> print('Hello, World!')"
    result = doctest(doc)
    expected = "```python\n>>> print('Hello, World!')\n```"
    assert result == expected

def test_doctest_with_multiple_lines():
    doc = ">>> print('Hello, World!')\n>>> print('Goodbye, World!')"
    result = doctest(doc)
    expected = "```python\n>>> print('Hello, World!')\n>>> print('Goodbye, World!')\n```"
    assert result == expected

def test_doctest_with_mixed_lines():
    doc = "Some text\n>>> print('Hello, World!')\nMore text"
    result = doctest(doc)
    expected = "Some text\n```python\n>>> print('Hello, World!')\n```\nMore text"
    assert result == expected

def test_doctest_with_no_doctest_lines():
    doc = "Some text\nMore text"
    result = doctest(doc)
    expected = "Some text\nMore text"
    assert result == expected

def test_doctest_with_multiple_doctest_blocks():
    doc = "Some text\n>>> print('Hello, World!')\nMore text\n>>> print('Goodbye, World!')"
    result = doctest(doc)
    expected = "Some text\n```python\n>>> print('Hello, World!')\n```\nMore text\n```python\n>>> print('Goodbye, World!')\n```"
    assert result == expected

def test_doctest_with_doctest_at_end():
    doc = "Some text\n>>> print('Hello, World!')"
    result = doctest(doc)
    expected = "Some text\n```python\n>>> print('Hello, World!')\n```"
    assert result == expected

def test_doctest_with_doctest_at_start():
    doc = ">>> print('Hello, World!')\nSome text"
    result = doctest(doc)
    expected = "```python\n>>> print('Hello, World!')\n```\nSome text"
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_api_with_classdef_node_and_empty_decorators():
    parser = Parser()
    class_node = ClassDef(name='TestClass', bases=[], keywords=[], body=[], decorator_list=[])
    parser.api('root', class_node, prefix='')
    assert not isinstance(class_node, (FunctionDef, AsyncFunctionDef))


# LLM-generated content at query #6
#--------------------------

```
def test_imports_simple_import():
    p = Parser()
    p.parse('test_module', "import os")
    assert p.alias == {'test_module.os': 'os'}

def test_imports_import_with_alias():
    p = Parser()
    p.parse('test_module', "import os as operating_system")
    assert p.alias == {'test_module.operating_system': 'os'}

def test_imports_from_import():
    p = Parser()
    p.parse('test_module', "from sys import path")
    assert p.alias == {'test_module.path': 'sys.path'}

def test_imports_from_import_with_alias():
    p = Parser()
    p.parse('test_module', "from sys import path as sys_path")
    assert p.alias == {'test_module.sys_path': 'sys.path'}

def test_imports_from_import_with_dots():
    p = Parser()
    p.parse('test_module', "from os.path import join")
    assert p.alias == {'test_module.join': 'os.path.join'}

def test_imports_relative_import():
    p = Parser()
    p.parse('test.module', "from ..sub import func")
    assert p.alias == {'test.module.func': 'test.sub.func'}

def test_imports_multiple_imports():
    p = Parser()
    p.parse('test_module', "import os, sys as system")
    assert p.alias == {'test_module.os': 'os', 'test_module.system': 'sys'}

def test_imports_multiple_from_imports():
    p = Parser()
    p.parse('test_module', "from sys import path, argv as arguments")
    assert p.alias == {'test_module.path': 'sys.path', 'test_module.arguments': 'sys.argv'}


# LLM-generated content at query #7
#--------------------------

```python
def test_compile_without_toc():
    parser = Parser(toc=False)
    parser.doc = {"module_name": "## Module `module_name`\n\n"}
    parser.docstring = {"module_name": "Docstring for module_name"}
    parser.const = {}
    parser.level = {"module_name": 0}
    parser.root = {"module_name": "module_name"}
    parser.imp = {"module_name": set()}
    assert parser.compile() == "## Module `module_name`\n\nDocstring for module_name\n"

def test_compile_with_toc():
    parser = Parser(toc=True)
    parser.doc = {"module_name": "## Module `module_name`\n\n"}
    parser.docstring = {"module_name": "Docstring for module_name"}
    parser.const = {}
    parser.level = {"module_name": 0}
    parser.root = {"module_name": "module_name"}
    parser.imp = {"module_name": set()}
    assert parser.compile() == "**Table of contents:**\n+ [module_name](#module-name)\n\n## Module `module_name`\n\nDocstring for module_name\n"

def test_compile_with_constants():
    parser = Parser(toc=False)
    parser.doc = {"module_name": "## Module `module_name`\n\n"}
    parser.docstring = {"module_name": "Docstring for module_name"}
    parser.const = {"module_name.CONST": "int"}
    parser.level = {"module_name": 0, "module_name.CONST": 1}
    parser.root = {"module_name": "module_name", "module_name.CONST": "module_name"}
    parser.imp = {"module_name": set()}
    assert parser.compile() == "## Module `module_name`\n\n| Constants | Type |\n|-----------|------|\n| CONST     | int  |\n\nDocstring for module_name\n"

def test_compile_with_missing_docstring():
    parser = Parser(toc=False)
    parser.doc = {"module_name": "## Module `module_name`\n\n"}
    parser.docstring = {}
    parser.const = {}
    parser.level = {"module_name": 0}
    parser.root = {"module_name": "module_name"}
    parser.imp = {"module_name": set()}
    assert parser.compile() == "## Module `module_name`\n\n"

def test_compile_with_non_public_name():
    parser = Parser(toc=False)
    parser.doc = {"module_name._private": "## Module `module_name._private`\n\n"}
    parser.docstring = {"module_name._private": "Docstring for module_name._private"}
    parser.const = {}
    parser.level = {"module_name._private": 1}
    parser.root = {"module_name._private": "module_name"}
    parser.imp = {"module_name": set()}
    assert parser.compile() == ""


# LLM-generated content at query #8
#--------------------------

```python
def test_load_docstring_with_valid_module():
    import types
    module = types.ModuleType('test_module')
    module.test_func = lambda: None
    module.test_func.__doc__ = "Test docstring"
    p = Parser()
    p.doc['test_module.test_func'] = "# test_func()\n\n*Full name:* `test_module.test_func`\n\n"
    p.load_docstring('test_module', module)
    assert p.docstring['test_module.test_func'] == "```python\nTest docstring\n```"

def test_load_docstring_with_none_docstring():
    import types
    module = types.ModuleType('test_module')
    module.test_func = lambda: None
    p = Parser()
    p.doc['test_module.test_func'] = "# test_func()\n\n*Full name:* `test_module.test_func`\n\n"
    p.load_docstring('test_module', module)
    assert 'test_module.test_func' not in p.docstring

def test_load_docstring_with_nested_attr():
    import types
    module = types.ModuleType('test_module')
    module.sub = types.ModuleType('sub')
    module.sub.test_func = lambda: None
    module.sub.test_func.__doc__ = "Nested docstring"
    p = Parser()
    p.doc['test_module.sub.test_func'] = "# test_func()\n\n*Full name:* `test_module.sub.test_func`\n\n"
    p.load_docstring('test_module', module)
    assert p.docstring['test_module.sub.test_func'] == "```python\nNested docstring\n```"

def test_load_docstring_with_non_existing_attr():
    import types
    module = types.ModuleType('test_module')
    p = Parser()
    p.doc['test_module.non_existing'] = "# non_existing()\n\n*Full name:* `test_module.non_existing`\n\n"
    p.load_docstring('test_module', module)
    assert 'test_module.non_existing' not in p.docstring


# LLM-generated content at query #9
#--------------------------

```
def test_is_public_returns_true_for_public_name():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == True

def test_is_public_returns_false_for_private_name():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root._name'] = 'root'
    assert parser.is_public('root._name') == False

def test_is_public_returns_true_for_name_in_all_list():
    parser = Parser()
    parser.imp['root'] = {'root.name'}
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == True

def test_is_public_returns_false_for_name_not_in_all_list():
    parser = Parser()
    parser.imp['root'] = {'root.other'}
    parser.root['root.name'] = 'root'
    assert parser.is_public('root.name') == False

def test_is_public_returns_true_for_root_module():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root'] = 'root'
    assert parser.is_public('root') == True

def test_is_public_returns_true_for_magic_name():
    parser = Parser()
    parser.imp['root'] = set()
    parser.root['root.__name__'] = 'root'
    assert parser.is_public('root.__name__') == True


# LLM-generated content at query #10
#--------------------------

```python
class TestClass:
    def __init__(self, value):
        self.value = value

def test__attr_single_level():
    obj = TestClass(42)
    assert _attr(obj, 'value') == 42

def test__attr_nested_level():
    class TestClass2:
        def __init__(self, inner_obj):
            self.inner = inner_obj
    obj = TestClass2(TestClass(42))
    assert _attr(obj, 'inner.value') == 42

def test__attr_nonexistent_single_level():
    obj = TestClass(42)
    assert _attr(obj, 'nonexistent') is None

def test__attr_nonexistent_nested_level():
    class TestClass2:
        def __init__(self, inner_obj):
            self.inner = inner_obj
    obj = TestClass2(TestClass(42))
    assert _attr(obj, 'inner.nonexistent') is None

def test__attr_nonexistent_intermediate_level():
    class TestClass2:
        def __init__(self, inner_obj):
            self.inner = inner_obj
    obj = TestClass2(TestClass(42))
    assert _attr(obj, 'nonexistent.value') is None


# LLM-generated content at query #11
#--------------------------

```
def test_walk_body_with_empty_body():
    body = []
    result = list(walk_body(body))
    assert result == []

def test_walk_body_with_single_node():
    body = [stmt()]
    result = list(walk_body(body))
    assert result == body

def test_walk_body_with_if_node():
    if_node = If(body=[stmt()], orelse=[stmt()])
    body = [if_node]
    result = list(walk_body(body))
    assert result == if_node.body + if_node.orelse

def test_walk_body_with_try_node():
    try_node = Try(body=[stmt()], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    body = [try_node]
    result = list(walk_body(body))
    expected = try_node.body + try_node.handlers[0].body + try_node.orelse + try_node.finalbody
    assert result == expected

def test_walk_body_with_mixed_nodes():
    node1 = stmt()
    if_node = If(body=[stmt()], orelse=[stmt()])
    node2 = stmt()
    try_node = Try(body=[stmt()], handlers=[ExceptHandler(body=[stmt()])], orelse=[stmt()], finalbody=[stmt()])
    node3 = stmt()
    body = [node1, if_node, node2, try_node, node3]
    result = list(walk_body(body))
    expected = [node1] + if_node.body + if_node.orelse + [node2] + try_node.body + try_node.handlers[0].body + try_node.orelse + try_node.finalbody + [node3]
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```
def test_class_api_with_bases_and_members():
    p = Parser()
    p.parse("test_module", "class A: pass")
    bases = [parse("B").body[0].value]
    body = [parse("x: int").body[0], parse("y = 1").body[0]]
    p.class_api("test_module", "test_module.A", bases, body)
    assert "Bases" in p.doc["test_module.A"]
    assert "Members" in p.doc["test_module.A"]
    assert "x" in p.doc["test_module.A"]
    assert "int" in p.doc["test_module.A"]
    assert "y" in p.doc["test_module.A"]

def test_class_api_with_enums():
    p = Parser()
    p.parse("test_module", "class A(enum.Enum): pass")
    bases = [parse("enum.Enum").body[0].value]
    body = [parse("X = 1").body[0], parse("Y = 2").body[0]]
    p.class_api("test_module", "test_module.A", bases, body)
    assert "Enums" in p.doc["test_module.A"]
    assert "X" in p.doc["test_module.A"]
    assert "Y" in p.doc["test_module.A"]

def test_class_api_with_no_bases_or_members():
    p = Parser()
    p.parse("test_module", "class A: pass")
    bases = []
    body = []
    p.class_api("test_module", "test_module.A", bases, body)
    assert "Bases" not in p.doc["test_module.A"]
    assert "Members" not in p.doc["test_module.A"]
    assert "Enums" not in p.doc["test_module.A"]

def test_class_api_with_private_members():
    p = Parser()
    p.parse("test_module", "class A: pass")
    bases = []
    body = [parse("_x: int").body[0], parse("__y = 1").body[0]]
    p.class_api("test_module", "test_module.A", bases, body)
    assert "Members" not in p.doc["test_module.A"]

def test_class_api_with_deleted_members():
    p = Parser()
    p.parse("test_module", "class A: pass")
    bases = []
    body = [parse("x = 1").body[0], parse("del x").body[0]]
    p.class_api("test_module", "test_module.A", bases, body)
    assert "Members" not in p.doc["test_module.A"]


# LLM-generated content at query #13
#--------------------------

```python
def test_load_docstring_with_non_none_doc():
    parser = Parser()
    parser.doc = {"test_module.func": "test doc"}
    class MockModule:
        def func(self):
            """test docstring"""
    mock_module = MockModule()
    parser.load_docstring("test_module", mock_module)
    assert parser.docstring["test_module.func"] == doctest("test docstring")


# LLM-generated content at query #14
#--------------------------

```python
def test_parse_method_with_empty_script():
    p = Parser()
    p.parse('empty_module', '')
    assert p.doc == {'empty_module': '## Module `empty_module`\n\n'}
    assert p.level == {'empty_module': 0}
    assert p.imp == {'empty_module': set()}
    assert p.root == {'empty_module': 'empty_module'}
    assert p.docstring == {}

def test_parse_method_with_import_statement():
    p = Parser()
    script = 'import os\nimport sys'
    p.parse('import_module', script)
    assert p.doc == {'import_module': '## Module `import_module`\n\n'}
    assert p.level == {'import_module': 0}
    assert p.imp == {'import_module': set()}
    assert p.root == {'import_module': 'import_module'}
    assert p.docstring == {}

def test_parse_method_with_function_def():
    p = Parser()
    script = 'def foo():\n    pass'
    p.parse('func_module', script)
    assert p.doc == {
        'func_module': '## Module `func_module`\n\n',
        'func_module.foo': '### foo()\n\n*Full name:* `func_module.foo`\n\n'
    }
    assert p.level == {'func_module': 0, 'func_module.foo': 0}
    assert p.imp == {'func_module': set()}
    assert p.root == {'func_module': 'func_module', 'func_module.foo': 'func_module'}
    assert p.docstring == {}

def test_parse_method_with_class_def():
    p = Parser()
    script = 'class Bar:\n    pass'
    p.parse('class_module', script)
    assert p.doc == {
        'class_module': '## Module `class_module`\n\n',
        'class_module.Bar': '### class Bar\n\n*Full name:* `class_module.Bar`\n\n'
    }
    assert p.level == {'class_module': 0, 'class_module.Bar': 0}
    assert p.imp == {'class_module': set()}
    assert p.root == {'class_module': 'class_module', 'class_module.Bar': 'class_module'}
    assert p.docstring == {}

def test_parse_method_with_docstring():
    p = Parser()
    script = '"""Module docstring"""'
    p.parse('doc_module', script)
    assert p.doc == {'doc_module': '## Module `doc_module`\n\n'}
    assert p.level == {'doc_module': 0}
    assert p.imp == {'doc_module': set()}
    assert p.root == {'doc_module': 'doc_module'}
    assert p.docstring == {'doc_module': '"""Module docstring"""'}


# LLM-generated content at query #15
#--------------------------

```python
def test_walk_body_yields_non_control_flow_nodes():
    body = [
        ast.Assign(targets=[ast.Name(id='x')], value=ast.Num(n=1)),
        ast.If(
            test=ast.Name(id='cond'),
            body=[ast.Assign(targets=[ast.Name(id='y')], value=ast.Num(n=2))],
            orelse=[]
        ),
        ast.Try(
            body=[ast.Assign(targets=[ast.Name(id='z')], value=ast.Num(n=3))],
            handlers=[],
            orelse=[],
            finalbody=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 3
    assert isinstance(result[0], ast.Assign)
    assert isinstance(result[1], ast.Assign)
    assert isinstance(result[2], ast.Assign)


# LLM-generated content at query #16
#--------------------------

```python
def test_is_magic_and_has_docstring():
    parser = Parser()
    parser.docstring = {'__magic__': 'docstring'}
    assert parser.compile() != ""


# LLM-generated content at query #17
#--------------------------

```python
def test_class_api_with_empty_bases_and_body():
    parser = Parser()
    parser.class_api('root', 'name', [], [])
    assert parser.doc['name'] == '### class name\n\n*Full name:* `name`\n\n'

def test_class_api_with_bases():
    parser = Parser()
    parser.class_api('root', 'name', [Constant('base1'), Constant('base2')], [])
    assert parser.doc['name'] == '### class name\n\n*Full name:* `name`\n\n| Bases |\n|:---:|\n| `base1` |\n| `base2` |\n\n'

def test_class_api_with_enums():
    parser = Parser()
    parser.class_api('root', 'name', [Constant('enum.Enum')], [AnnAssign(Name('attr1'), None, None), AnnAssign(Name('attr2'), None, None)])
    assert parser.doc['name'] == '### class name\n\n*Full name:* `name`\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n\n| Enums |\n|:---:|\n| attr1 |\n| attr2 |\n\n'

def test_class_api_with_members():
    parser = Parser()
    parser.class_api('root', 'name', [], [AnnAssign(Name('attr1'), None, Constant('type1')), AnnAssign(Name('attr2'), None, Constant('type2'))])
    assert parser.doc['name'] == '### class name\n\n*Full name:* `name`\n\n| Members | Type |\n|:---:|:---:|\n| `attr1` | `type1` |\n| `attr2` | `type2` |\n\n'

def test_class_api_with_delete_statement():
    parser = Parser()
    parser.class_api('root', 'name', [], [AnnAssign(Name('attr1'), None, Constant('type1')), Delete([Name('attr1')])])
    assert parser.doc['name'] == '### class name\n\n*Full name:* `name`\n\n'


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    parser = Parser()
    result = isinstance(node, Assign) and len(node.targets) == 1 and isinstance(node.targets[0], Name)
    assert result is False


# LLM-generated content at query #19
#--------------------------

```python
def test_compile_with_toc():
    parser = Parser(toc=True)
    parser.doc = {
        'module': '### Module `module`\n\n',
        'module.func': '#### func()\n\n*Full name:* `module.func`\n\n'
    }
    parser.docstring = {
        'module.func': 'Function docstring'
    }
    parser.level = {
        'module': 0,
        'module.func': 1
    }
    parser.root = {
        'module': 'module',
        'module.func': 'module'
    }
    parser.imp = {
        'module': set()
    }
    parser.const = {}
    result = parser.compile()
    expected = '**Table of contents:**\n    + [`module`](#module)\n        + [`module.func`](#module-func)\n\n### Module `module`\n\n\n#### func()\n\n*Full name:* `module.func`\n\nFunction docstring\n'
    assert result == expected

def test_compile_without_toc():
    parser = Parser(toc=False)
    parser.doc = {
        'module': '### Module `module`\n\n',
        'module.func': '#### func()\n\n*Full name:* `module.func`\n\n'
    }
    parser.docstring = {
        'module.func': 'Function docstring'
    }
    parser.level = {
        'module': 0,
        'module.func': 1
    }
    parser.root = {
        'module': 'module',
        'module.func': 'module'
    }
    parser.imp = {
        'module': set()
    }
    parser.const = {}
    result = parser.compile()
    expected = '### Module `module`\n\n\n#### func()\n\n*Full name:* `module.func`\n\nFunction docstring\n'
    assert result == expected

def test_compile_with_constants():
    parser = Parser(toc=False)
    parser.doc = {
        'module': '### Module `module`\n\n'
    }
    parser.docstring = {}
    parser.level = {
        'module': 0
    }
    parser.root = {
        'module': 'module'
    }
    parser.imp = {
        'module': set()
    }
    parser.const = {
        'module.CONST': 'int'
    }
    result = parser.compile()
    expected = '### Module `module`\n\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n'
    assert result == expected

def test_compile_with_missing_docstring_warning():
    parser = Parser(toc=False)
    parser.doc = {
        'module': '### Module `module`\n\n',
        'module.__magic__': '#### __magic__()\n\n*Full name:* `module.__magic__`\n\n'
    }
    parser.docstring = {}
    parser.level = {
        'module': 0,
        'module.__magic__': 1
    }
    parser.root = {
        'module': 'module',
        'module.__magic__': 'module'
    }
    parser.imp = {
        'module': set()
    }
    parser.const = {}
    result = parser.compile()
    expected = '### Module `module`\n\n\n#### __magic__()\n\n*Full name:* `module.__magic__`\n\n'
    assert result == expected


# LLM-generated content at query #20
#--------------------------

```
def test_api_method_creates_function_doc():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[])
    parser.api(root, node)
    assert "test_func()" in parser.doc["test_module.test_func"]
    assert "*Full name:* `test_module.test_func`" in parser.doc["test_module.test_func"]

def test_api_method_creates_async_function_doc():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[])
    parser.api(root, node)
    assert "async test_async_func()" in parser.doc["test_module.test_async_func"]
    assert "*Full name:* `test_module.test_async_func`" in parser.doc["test_module.test_async_func"]

def test_api_method_creates_class_doc():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    assert "*Full name:* `test_module.TestClass`" in parser.doc["test_module.TestClass"]

def test_api_method_includes_decorators():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[Name(id="decorator", ctx=Load())])
    parser.api(root, node)
    assert "Decorators" in parser.doc["test_module.test_func"]
    assert "@decorator" in parser.doc["test_module.test_func"]

def test_api_method_includes_docstring():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[Expr(value=Constant(value="Test docstring"))], decorator_list=[])
    parser.api(root, node)
    assert "Test docstring" in parser.docstring["test_module.test_func"]

def test_api_method_nested_class():
    parser = Parser()
    root = "test_module"
    outer_class = ClassDef(name="OuterClass", bases=[], keywords=[], body=[], decorator_list=[])
    inner_func = FunctionDef(name="inner_func", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[])
    outer_class.body.append(inner_func)
    parser.api(root, outer_class)
    assert "test_module.OuterClass.inner_func" in parser.doc
    assert "inner_func()" in parser.doc["test_module.OuterClass.inner_func"]


# LLM-generated content at query #21
#--------------------------

```python
def test_visit_Constant_returns_node_for_non_string_constant():
    resolver = Resolver("root", {})
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_returns_node_for_invalid_string_syntax():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax @#$%")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_visits_parsed_expression_for_valid_string():
    resolver = Resolver("root", {})
    node = Constant("some_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "some_name"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #22
#--------------------------

```python
def test_is_public_returns_true_for_public_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root.name': 'root'}
    assert parser.is_public('root.name') == True

def test_is_public_returns_false_for_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root._name': 'root'}
    assert parser.is_public('root._name') == False

def test_is_public_returns_true_for_name_in_all():
    parser = Parser()
    parser.imp = {'root': {'root.name'}}
    parser.root = {'root.name': 'root'}
    assert parser.is_public('root.name') == True

def test_is_public_returns_true_for_root_in_all():
    parser = Parser()
    parser.imp = {'root': {'root'}}
    parser.root = {'root': 'root'}
    assert parser.is_public('root') == True

def test_is_public_returns_false_for_non_public_name_not_in_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root.other': 'root'}
    assert parser.is_public('root.name') == False


# LLM-generated content at query #23
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    parser.doc = {}
    parser.const = {}
    parser.root = {'root': 'root'}
    parser.imp = {'root': set()}
    parser.level = {'root': 0}
    parser.alias = {}
    parser.resolve = lambda root, node: unparse(node)
    node = ClassDef(name='MyClass', bases=[Name(id='BaseClass')], body=[], decorator_list=[])
    parser.class_api('root', 'root.MyClass', [Name(id='BaseClass')], [])
    assert 'root.MyClass' in parser.doc
    assert parser.doc['root.MyClass'] == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'

def test_class_api_with_enums():
    parser = Parser()
    parser.doc = {}
    parser.const = {}
    parser.root = {'root': 'root'}
    parser.imp = {'root': set()}
    parser.level = {'root': 0}
    parser.alias = {}
    parser.resolve = lambda root, node: unparse(node)
    node = ClassDef(name='MyEnum', bases=[Name(id='Enum')], body=[AnnAssign(target=Name(id='A'), annotation=Name(id='int'))], decorator_list=[])
    parser.class_api('root', 'root.MyEnum', [Name(id='Enum')], [AnnAssign(target=Name(id='A'), annotation=Name(id='int'))])
    assert 'root.MyEnum' in parser.doc
    assert parser.doc['root.MyEnum'] == '### class MyEnum\n\n*Full name:* `root.MyEnum`\n\n'

def test_class_api_with_members():
    parser = Parser()
    parser.doc = {}
    parser.const = {}
    parser.root = {'root': 'root'}
    parser.imp = {'root': set()}
    parser.level = {'root': 0}
    parser.alias = {}
    parser.resolve = lambda root, node: unparse(node)
    node = ClassDef(name='MyClass', bases=[], body=[AnnAssign(target=Name(id='attr'), annotation=Name(id='int'))], decorator_list=[])
    parser.class_api('root', 'root.MyClass', [], [AnnAssign(target=Name(id='attr'), annotation=Name(id='int'))])
    assert 'root.MyClass' in parser.doc
    assert parser.doc['root.MyClass'] == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'

def test_class_api_with_no_bases_or_members():
    parser = Parser()
    parser.doc = {}
    parser.const = {}
    parser.root = {'root': 'root'}
    parser.imp = {'root': set()}
    parser.level = {'root': 0}
    parser.alias = {}
    parser.resolve = lambda root, node: unparse(node)
    node = ClassDef(name='MyClass', bases=[], body=[], decorator_list=[])
    parser.class_api('root', 'root.MyClass', [], [])
    assert 'root.MyClass' in parser.doc
    assert parser.doc['root.MyClass'] == '### class MyClass\n\n*Full name:* `root.MyClass`\n\n'


# LLM-generated content at query #24
#--------------------------

```python
def test_visit_Attribute_with_typing_prefix():
    resolver = Resolver("root", {"alias": "value"})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Attribute_without_typing_prefix():
    resolver = Resolver("root", {"alias": "value"})
    node = Attribute(value=Name(id="module", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "module"
    assert result.attr == "List"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_func_api_with_posonlyargs():
    p = Parser()
    args = [arg('a', None), arg('b', None)]
    defaults = [None, None]
    p.func_api('root', 'name', arguments(posonlyargs=args, args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| a | b | return |\n|:---:|:---:|:---:|\n|  |  |  |\n\n"

def test_func_api_with_vararg():
    p = Parser()
    args = [arg('*args', None)]
    p.func_api('root', 'name', arguments(posonlyargs=[], args=[], vararg=args[0], kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| *args | return |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_kwarg():
    p = Parser()
    args = [arg('**kwargs', None)]
    p.func_api('root', 'name', arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=args[0], defaults=[]), None, has_self=False, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| **kwargs | return |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_has_self():
    p = Parser()
    args = [arg('self', None), arg('a', None)]
    p.func_api('root', 'name', arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), None, has_self=True, cls_method=False)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| self | a | return |\n|:---:|:---:|:---:|\n| Self |  |  |\n\n"

def test_func_api_with_cls_method():
    p = Parser()
    args = [arg('cls', None), arg('a', None)]
    p.func_api('root', 'name', arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), None, has_self=True, cls_method=True)
    assert p.doc['name'] == "### name()\n\n*Full name:* `name`\n\n| cls | a | return |\n|:---:|:---:|:---:|\n| type[Self] |  |  |\n\n"


# LLM-generated content at query #2
#--------------------------

```
def test_compile_with_toc():
    p = Parser(toc=True)
    p.doc = {'module': '## Module `module`\n<a id="module"></a>\n\n'}
    p.docstring = {'module': 'Module docstring'}
    p.imp = {'module': set()}
    p.root = {'module': 'module'}
    p.level = {'module': 0}
    p.const = {}
    result = p.compile()
    expected = '**Table of contents:**\n    + [`module`](#module)\n\n## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert result == expected

def test_compile_without_toc():
    p = Parser(toc=False)
    p.doc = {'module': '## Module `module`\n<a id="module"></a>\n\n'}
    p.docstring = {'module': 'Module docstring'}
    p.imp = {'module': set()}
    p.root = {'module': 'module'}
    p.level = {'module': 0}
    p.const = {}
    result = p.compile()
    expected = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert result == expected

def test_compile_with_constants():
    p = Parser(toc=False)
    p.doc = {'module': '## Module `module`\n<a id="module"></a>\n\n'}
    p.docstring = {'module': 'Module docstring'}
    p.imp = {'module': set()}
    p.root = {'module': 'module', 'module.CONST': 'module'}
    p.level = {'module': 0, 'module.CONST': 1}
    p.const = {'module.CONST': 'int'}
    result = p.compile()
    expected = '## Module `module`\n<a id="module"></a>\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n\nModule docstring\n'
    assert result == expected

def test_compile_with_non_public_members():
    p = Parser(toc=False)
    p.doc = {'module': '## Module `module`\n<a id="module"></a>\n\n', 'module._private': '### _private()\n\n*Full name:* `module._private`\n<a id="module._private"></a>\n\n'}
    p.docstring = {'module': 'Module docstring', 'module._private': 'Private function'}
    p.imp = {'module': set()}
    p.root = {'module': 'module', 'module._private': 'module'}
    p.level = {'module': 0, 'module._private': 1}
    p.const = {}
    result = p.compile()
    expected = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert result == expected

def test_compile_with_magic_methods():
    p = Parser(toc=False)
    p.doc = {'module': '## Module `module`\n<a id="module"></a>\n\n', 'module.__magic__': '### __magic__()\n\n*Full name:* `module.__magic__`\n<a id="module.__magic__"></a>\n\n'}
    p.docstring = {'module': 'Module docstring', 'module.__magic__': 'Magic method'}
    p.imp = {'module': set()}
    p.root = {'module': 'module', 'module.__magic__': 'module'}
    p.level = {'module': 0, 'module.__magic__': 1}
    p.const = {}
    result = p.compile()
    expected = '## Module `module`\n<a id="module"></a>\n\nModule docstring\n'
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_const_type_constant_int():
    node = Constant(42)
    assert const_type(node) == 'int'

def test_const_type_constant_str():
    node = Constant('hello')
    assert const_type(node) == 'str'

def test_const_type_tuple_empty():
    node = Tuple([], None)
    assert const_type(node) == 'tuple[]'

def test_const_type_tuple_int():
    node = Tuple([Constant(1), Constant(2)], None)
    assert const_type(node) == 'tuple[int, int]'

def test_const_type_tuple_mixed_types():
    node = Tuple([Constant(1), Constant('hello')], None)
    assert const_type(node) == 'tuple[Any, Any]'

def test_const_type_list_empty():
    node = List([], None)
    assert const_type(node) == 'list[]'

def test_const_type_list_str():
    node = List([Constant('a'), Constant('b')], None)
    assert const_type(node) == 'list[str, str]'

def test_const_type_set_empty():
    node = Set([])
    assert const_type(node) == 'set[]'

def test_const_type_set_float():
    node = Set([Constant(1.0), Constant(2.0)])
    assert const_type(node) == 'set[float, float]'

def test_const_type_dict_empty():
    node = Dict([], [])
    assert const_type(node) == 'dict[]'

def test_const_type_dict_str_int():
    node = Dict([Constant('key')], [Constant(1)])
    assert const_type(node) == 'dict[str, int]'

def test_const_type_call_builtin_int():
    node = Call(Name('int'), [Constant('42')], [])
    assert const_type(node) == 'int'

def test_const_type_call_builtin_str():
    node = Call(Name('str'), [Constant(42)], [])
    assert const_type(node) == 'str'

def test_const_type_call_non_builtin():
    node = Call(Name('custom_func'), [], [])
    assert const_type(node) == 'Any'

def test_const_type_unsupported_node():
    node = Name('unsupported')
    assert const_type(node) == 'Any'


# LLM-generated content at query #4
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link == False
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link == True

def test_parser_constructor_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link == True


# LLM-generated content at query #5
#--------------------------

```python
def test_is_public_predicate_evaluates_to_true():
    parser = Parser()
    parser.doc = {"module.name": "some_doc"}
    parser.root = {"module.name": "module"}
    parser.level = {"module.name": 1}
    parser.imp = {"module": set()}
    parser.const = {}
    assert parser.is_public("module.name") == True


# LLM-generated content at query #6
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='object', ctx=Load())], [])
    assert 'Bases' in parser.doc['test_module.A']
    assert 'object' in parser.doc['test_module.A']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [])
    assert 'Bases' in parser.doc['test_module.A']
    assert 'enum.Enum' in parser.doc['test_module.A']

def test_class_api_with_members():
    parser = Parser()
    parser.parse('test_module', 'class A: x: int = 1')
    node = parse('class A: x: int = 1').body[0]
    parser.class_api('test_module', 'test_module.A', [], node.body)
    assert 'Members' in parser.doc['test_module.A']
    assert 'x' in parser.doc['test_module.A']
    assert 'int' in parser.doc['test_module.A']

def test_class_api_with_enums():
    parser = Parser()
    parser.parse('test_module', 'class A(enum.Enum): X = 1')
    node = parse('class A(enum.Enum): X = 1').body[0]
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], node.body)
    assert 'Enums' in parser.doc['test_module.A']
    assert 'X' in parser.doc['test_module.A']

def test_class_api_with_private_members():
    parser = Parser()
    parser.parse('test_module', 'class A: _x: int = 1')
    node = parse('class A: _x: int = 1').body[0]
    parser.class_api('test_module', 'test_module.A', [], node.body)
    assert 'Members' not in parser.doc['test_module.A']
    assert '_x' not in parser.doc['test_module.A']

def test_class_api_with_deleted_members():
    parser = Parser()
    parser.parse('test_module', 'class A: x: int = 1; del x')
    node = parse('class A: x: int = 1; del x').body[0]
    parser.class_api('test_module', 'test_module.A', [], node.body)
    assert 'Members' not in parser.doc['test_module.A']
    assert 'x' not in parser.doc['test_module.A']


# LLM-generated content at query #7
#--------------------------

```python
def test_parser_constructor_default_values():
    p = Parser()
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_constructor_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link == False
    assert p.b_level == 2
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #8
#--------------------------

```
def test_is_public_with_public_name():
    p = Parser()
    p.imp['root'] = set()
    p.root['root.public_name'] = 'root'
    assert p.is_public('root.public_name') == True

def test_is_public_with_private_name():
    p = Parser()
    p.imp['root'] = set()
    p.root['root._private_name'] = 'root'
    assert p.is_public('root._private_name') == False

def test_is_public_with_magic_name():
    p = Parser()
    p.imp['root'] = set()
    p.root['root.__magic__'] = 'root'
    assert p.is_public('root.__magic__') == False

def test_is_public_with_name_in_all():
    p = Parser()
    p.imp['root'] = {'root.public_name'}
    p.root['root.public_name'] = 'root'
    assert p.is_public('root.public_name') == True

def test_is_public_with_parent_in_all():
    p = Parser()
    p.imp['root'] = {'root'}
    p.root['root.public_name'] = 'root'
    assert p.is_public('root.public_name') == True

def test_is_public_with_root_module():
    p = Parser()
    p.imp['root'] = set()
    p.root['root'] = 'root'
    assert p.is_public('root') == True


# LLM-generated content at query #9
#--------------------------

```
def test__defaults_with_non_none_args():
    args = [1, 2, 3]
    result = list(_defaults(args))
    assert result == ["`1`", "`2`", "`3`"]

def test__defaults_with_none_args():
    args = [1, None, 3]
    result = list(_defaults(args))
    assert result == ["`1`", " ", "`3`"]

def test__defaults_with_empty_args():
    args = []
    result = list(_defaults(args))
    assert result == []

def test__defaults_with_all_none_args():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_ampersand_in_arg():
    args = ["a & b"]
    result = list(_defaults(args))
    assert result == ["<code>a &amp; b</code>"]

def test__defaults_with_pipe_in_arg():
    args = ["a | b"]
    result = list(_defaults(args))
    assert result == ["<code>a &#124; b</code>"]


# LLM-generated content at query #10
#--------------------------

```
def test_api_method_creates_function_doc():
    p = Parser()
    func_node = FunctionDef(name='test_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    p.api('root', func_node)
    assert 'test_func()' in p.doc['root.test_func']

def test_api_method_creates_async_function_doc():
    p = Parser()
    async_func_node = AsyncFunctionDef(name='test_async_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    p.api('root', async_func_node)
    assert 'async test_async_func()' in p.doc['root.test_async_func']

def test_api_method_creates_class_doc():
    p = Parser()
    class_node = ClassDef(name='TestClass', bases=[], keywords=[], body=[], decorator_list=[])
    p.api('root', class_node)
    assert 'class TestClass' in p.doc['root.TestClass']

def test_api_method_includes_decorators():
    p = Parser()
    func_node = FunctionDef(name='test_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[Name(id='decorator', ctx=Load())], returns=None)
    p.api('root', func_node)
    assert 'Decorators' in p.doc['root.test_func']

def test_api_method_includes_bases_for_class():
    p = Parser()
    class_node = ClassDef(name='TestClass', bases=[Name(id='BaseClass', ctx=Load())], keywords=[], body=[], decorator_list=[])
    p.api('root', class_node)
    assert 'Bases' in p.doc['root.TestClass']

def test_api_method_includes_members_for_class():
    p = Parser()
    assign_node = AnnAssign(target=Name(id='member', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)
    class_node = ClassDef(name='TestClass', bases=[], keywords=[], body=[assign_node], decorator_list=[])
    p.api('root', class_node)
    assert 'Members' in p.doc['root.TestClass']

def test_api_method_includes_enums_for_enum_class():
    p = Parser()
    assign_node = Assign(targets=[Name(id='ENUM_VALUE', ctx=Store())], value=Constant(value=1))
    class_node = ClassDef(name='TestEnum', bases=[Name(id='enum.Enum', ctx=Load())], keywords=[], body=[assign_node], decorator_list=[])
    p.api('root', class_node)
    assert 'Enums' in p.doc['root.TestEnum']

def test_api_method_includes_docstring():
    p = Parser()
    docstring = '"""Test docstring"""'
    func_node = FunctionDef(name='test_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Expr(value=Constant(value=docstring))], decorator_list=[], returns=None)
    p.api('root', func_node)
    assert 'Test docstring' in p.docstring['root.test_func']

def test_api_method_handles_nested_classes():
    p = Parser()
    inner_class_node = ClassDef(name='InnerClass', bases=[], keywords=[], body=[], decorator_list=[])
    outer_class_node = ClassDef(name='OuterClass', bases=[], keywords=[], body=[inner_class_node], decorator_list=[])
    p.api('root', outer_class_node)
    assert 'class InnerClass' in p.doc['root.OuterClass.InnerClass']

def test_api_method_handles_nested_functions():
    p = Parser()
    inner_func_node = FunctionDef(name='inner_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    outer_func_node = FunctionDef(name='outer_func', args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[inner_func_node], decorator_list=[], returns=None)
    p.api('root', outer_func_node)
    assert 'inner_func()' in p.doc['root.outer_func.inner_func']


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    root = "test_module"
    name = "test_class"
    body = [
        Assign(targets=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())], value=Constant(value=42)),
        Assign(targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())], value=Constant(value=(1, 2))),
        Assign(targets=[Subscript(value=Name(id="arr", ctx=Load()), slice=Constant(value=0), ctx=Store())], value=Constant(value=10))
    ]
    parser.class_api(root, name, [], body)
    assert "x" not in parser.doc[name]
    assert "y" not in parser.doc[name]
    assert "a" not in parser.doc[name]
    assert "b" not in parser.doc[name]
    assert "arr" not in parser.doc[name]


# LLM-generated content at query #12
#--------------------------

```python
def test_parse_sets_doc_and_level_correctly():
    parser = Parser()
    parser.parse("module_name", "import os")
    assert parser.doc["module_name"] == "# Module `module_name`\n\n"
    assert parser.level["module_name"] == 0

def test_parse_sets_root_and_imp_correctly():
    parser = Parser()
    parser.parse("module_name", "import os")
    assert parser.root["module_name"] == "module_name"
    assert parser.imp["module_name"] == set()

def test_parse_handles_docstring_correctly():
    parser = Parser()
    script = '"""This is a docstring."""'
    parser.parse("module_name", script)
    assert "This is a docstring." in parser.docstring["module_name"]

def test_parse_handles_imports_correctly():
    parser = Parser()
    parser.parse("module_name", "import os")
    assert "os" in parser.alias.values()

def test_parse_handles_assignments_correctly():
    parser = Parser()
    parser.parse("module_name", "x = 10")
    assert "x" in parser.alias.keys()

def test_parse_handles_function_definitions_correctly():
    parser = Parser()
    parser.parse("module_name", "def func(): pass")
    assert "func" in parser.alias.keys()

def test_parse_handles_class_definitions_correctly():
    parser = Parser()
    parser.parse("module_name", "class MyClass: pass")
    assert "MyClass" in parser.alias.keys()


# LLM-generated content at query #13
#--------------------------

```
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    p.globals('root', node)
    assert p.alias['root.x'] == '42'
    assert p.const['root.x'] == 'int'

def test_globals_with_assign():
    p = Parser()
    node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value='test'))
    p.globals('root', node)
    assert p.alias['root.y'] == "'test'"
    assert p.const['root.y'] == 'str'

def test_globals_with_assign_and_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14), type_comment='float')
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_with_non_name_target():
    p = Parser()
    node = Assign(targets=[Subscript(value=Name(id='arr', ctx=Load()), slice=Constant(value=0), ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.arr[0]' not in p.alias
    assert 'root.arr[0]' not in p.const

def test_globals_with_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert 'root.a' not in p.alias
    assert 'root.b' not in p.alias

def test_globals_with_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=100))
    p.globals('root', node)
    assert p.alias['root.CONST'] == '100'
    assert p.const['root.CONST'] == 'int'

def test_globals_with___all__():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='func1'), Constant(value='func2')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.func1', 'root.func2'}


# LLM-generated content at query #14
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "self_ty")
    node = Name(id="self_ty", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"
    assert isinstance(result.ctx, Load)

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias"}, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_alias_and_typevar():
    resolver = Resolver("root", {"root.name": "typing.TypeVar"}, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_without_alias():
    resolver = Resolver("root", {}, "")
    node = Name(id="name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #15
#--------------------------

```
def test_walk_body_empty_body():
    body = []
    result = list(walk_body(body))
    assert result == []

def test_walk_body_single_node():
    class MockNode:
        pass
    node = MockNode()
    body = [node]
    result = list(walk_body(body))
    assert result == [node]

def test_walk_body_if_statement():
    class MockNode:
        pass
    class If:
        def __init__(self, body, orelse):
            self.body = body
            self.orelse = orelse
    node1 = MockNode()
    node2 = MockNode()
    node3 = MockNode()
    if_node = If(body=[node2], orelse=[node3])
    body = [node1, if_node]
    result = list(walk_body(body))
    assert result == [node1, node2, node3]

def test_walk_body_try_statement():
    class MockNode:
        pass
    class Try:
        def __init__(self, body, handlers, orelse, finalbody):
            self.body = body
            self.handlers = handlers
            self.orelse = orelse
            self.finalbody = finalbody
    class Handler:
        def __init__(self, body):
            self.body = body
    node1 = MockNode()
    node2 = MockNode()
    node3 = MockNode()
    node4 = MockNode()
    node5 = MockNode()
    handler = Handler(body=[node3])
    try_node = Try(body=[node2], handlers=[handler], orelse=[node4], finalbody=[node5])
    body = [node1, try_node]
    result = list(walk_body(body))
    assert result == [node1, node2, node3, node4, node5]

def test_walk_body_nested_control_structures():
    class MockNode:
        pass
    class If:
        def __init__(self, body, orelse):
            self.body = body
            self.orelse = orelse
    class Try:
        def __init__(self, body, handlers, orelse, finalbody):
            self.body = body
            self.handlers = handlers
            self.orelse = orelse
            self.finalbody = finalbody
    class Handler:
        def __init__(self, body):
            self.body = body
    node1 = MockNode()
    node2 = MockNode()
    node3 = MockNode()
    node4 = MockNode()
    node5 = MockNode()
    node6 = MockNode()
    node7 = MockNode()
    if_node = If(body=[node3], orelse=[node4])
    handler = Handler(body=[node6])
    try_node = Try(body=[node2, if_node], handlers=[handler], orelse=[node5], finalbody=[node7])
    body = [node1, try_node]
    result = list(walk_body(body))
    assert result == [node1, node2, node3, node4, node6, node5, node7]


# LLM-generated content at query #16
#--------------------------

```python
def test_class_api_predicate_evaluates_false():
    parser = Parser()
    node = Assign(targets=[Name(id="attr", ctx=Store())], value=Constant(value=42), type_comment="int")
    parser.class_api("root", "name", [], [node])


# LLM-generated content at query #17
#--------------------------

```
def test_doctest_empty_input():
    assert doctest("") == ""

def test_doctest_no_doctest_lines():
    input_doc = "This is a normal docstring.\nWith multiple lines.\nNo doctest here."
    expected = "This is a normal docstring.\nWith multiple lines.\nNo doctest here."
    assert doctest(input_doc) == expected

def test_doctest_single_doctest_line():
    input_doc = ">>> print('hello')"
    expected = "```python\n>>> print('hello')\n```"
    assert doctest(input_doc) == expected

def test_doctest_multiple_doctest_lines():
    input_doc = ">>> x = 1\n>>> print(x)"
    expected = "```python\n>>> x = 1\n>>> print(x)\n```"
    assert doctest(input_doc) == expected

def test_doctest_mixed_content():
    input_doc = "Normal text\n>>> doctest line\nMore normal text"
    expected = "Normal text\n```python\n>>> doctest line\n```\nMore normal text"
    assert doctest(input_doc) == expected

def test_doctest_multiple_doctest_blocks():
    input_doc = "Text\n>>> block1\nText\n>>> block2\nText"
    expected = "Text\n```python\n>>> block1\n```\nText\n```python\n>>> block2\n```\nText"
    assert doctest(input_doc) == expected

def test_doctest_ends_with_doctest():
    input_doc = "Text\n>>> last line"
    expected = "Text\n```python\n>>> last line\n```"
    assert doctest(input_doc) == expected


# LLM-generated content at query #18
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='B', ctx=Load())], [])
    assert 'B' in parser.doc['test_module.A']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name(id='enum.Enum', ctx=Load())], [Assign(targets=[Name(id='X', ctx=Store())], value=Constant(value=1))])
    assert 'X' in parser.doc['test_module.A']

def test_class_api_with_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert 'x' in parser.doc['test_module.A']

def test_class_api_with_deleted_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None), Delete(targets=[Name(id='x', ctx=Del())])])
    assert 'x' not in parser.doc['test_module.A']

def test_class_api_with_private_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [], [AnnAssign(target=Name(id='_x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)])
    assert '_x' not in parser.doc['test_module.A']


# LLM-generated content at query #19
#--------------------------

```python
def test_visit_Attribute_with_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_without_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="other", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_with_non_name_value():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Constant(value="typing"), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_true():
    node = Try()
    body = [node]
    result = next(walk_body(body))
    assert isinstance(result, Try)


# LLM-generated content at query #21
#--------------------------

```python
def test_constructor_default_values():
    parser = Parser()
    assert parser.link == True
    assert parser.b_level == 1
    assert parser.toc == False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_constructor_custom_values():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link == False
    assert parser.b_level == 2
    assert parser.toc == True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_post_init_with_toc():
    parser = Parser(toc=True)
    assert parser.link == True

def test_post_init_without_toc():
    parser = Parser(toc=False)
    assert parser.link == True


# LLM-generated content at query #22
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name('B', Load())], [])
    assert 'Bases' in parser.doc['test_module.A']

def test_class_api_with_enum_bases():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    parser.class_api('test_module', 'test_module.A', [Name('enum.Enum', Load())], [])
    assert 'Enums' in parser.doc['test_module.A']

def test_class_api_with_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [AnnAssign(Name('x', Store()), Name('int', Load()), None, 1)]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert 'Members' in parser.doc['test_module.A']

def test_class_api_with_deleted_member():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [
        AnnAssign(Name('x', Store()), Name('int', Load()), None, 1),
        Delete([Name('x', Del())])
    ]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert 'Members' not in parser.doc['test_module.A']

def test_class_api_with_private_member():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [AnnAssign(Name('_x', Store()), Name('int', Load()), None, 1)]
    parser.class_api('test_module', 'test_module.A', [], body)
    assert 'Members' not in parser.doc['test_module.A']

def test_class_api_with_enum_members():
    parser = Parser()
    parser.parse('test_module', 'class A: pass')
    body = [Assign([Name('X', Store())], Constant(1), None)]
    parser.class_api('test_module', 'test_module.A', [Name('enum.Enum', Load())], body)
    assert 'Enums' in parser.doc['test_module.A']


# LLM-generated content at query #23
#--------------------------

```python
def test_func_api_has_default_true():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None)
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert 'items=[ann]' in parser.doc['name']

def test_func_api_has_default_false():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('test', None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None)
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert 'items=[ann, _defaults(default)]' in parser.doc['name']


# LLM-generated content at query #24
#--------------------------

```
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    assign_node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=42))
    result = isinstance(assign_node, Assign) and len(assign_node.targets) == 1 and isinstance(assign_node.targets[0], Name)
    assert result is False


# LLM-generated content at query #25
#--------------------------

```python
def test_func_ann():
    parser = Parser(link=True, b_level=1, toc=False)
    args = [
        arg(arg='self', annotation=Name(id='Self')),
        arg(arg='x', annotation=Name(id='int')),
        arg(arg='y', annotation=Name(id='str')),
        arg(arg='return', annotation=Name(id='bool'))
    ]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str', 'bool']


# LLM-generated content at query #26
#--------------------------

```
def test_api_method_with_function_def():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_function", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "test_function()" in parser.doc["test_module.test_function"]

def test_api_method_with_async_function_def():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_function", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "async test_async_function()" in parser.doc["test_module.test_async_function"]

def test_api_method_with_class_def():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "class TestClass" in parser.doc["test_module.TestClass"]

def test_api_method_with_prefix():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_method", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node, prefix="TestClass")
    assert "test_method()" in parser.doc["test_module.TestClass.test_method"]

def test_api_method_with_decorators():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_decorated_function", args=arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[Name(id="decorator", ctx=Load())], returns=None)
    parser.api(root, node)
    assert "@decorator" in parser.doc["test_module.test_decorated_function"]


# LLM-generated content at query #27
#--------------------------

```
def test_has_self_predicate_evaluates_to_true():
    parser = Parser()
    root = "test_root"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[])
    prefix = "test_prefix"
    parser.api(root, node, prefix=prefix)
    assert bool(prefix) and '@staticmethod' not in ['@' + parser.resolve(root, d) for d in node.decorator_list]


# LLM-generated content at query #28
#--------------------------

```
def test_class_api_with_bases():
    parser = Parser()
    parser.parse("test_module", "class A: pass")
    parser.class_api("test_module", "test_module.A", [], [])
    assert "Bases" not in parser.doc["test_module.A"]

    parser = Parser()
    parser.parse("test_module", "class B(A): pass")
    parser.class_api("test_module", "test_module.B", [Name("A", Load())], [])
    assert "Bases" in parser.doc["test_module.B"]
    assert "A" in parser.doc["test_module.B"]

def test_class_api_with_members():
    parser = Parser()
    parser.parse("test_module", "class C: x: int = 1")
    parser.class_api("test_module", "test_module.C", [], [
        AnnAssign(Name("x", Store()), Name("int", Load()), Constant(1), 1)
    ])
    assert "Members" in parser.doc["test_module.C"]
    assert "x" in parser.doc["test_module.C"]
    assert "int" in parser.doc["test_module.C"]

def test_class_api_with_enums():
    parser = Parser()
    parser.parse("test_module", "class D(enum.Enum): X = 1")
    parser.class_api("test_module", "test_module.D", [Name("enum.Enum", Load())], [
        Assign([Name("X", Store())], Constant(1), None)
    ])
    assert "Enums" in parser.doc["test_module.D"]
    assert "X" in parser.doc["test_module.D"]

def test_class_api_with_private_members():
    parser = Parser()
    parser.parse("test_module", "class E: _x: int = 1")
    parser.class_api("test_module", "test_module.E", [], [
        AnnAssign(Name("_x", Store()), Name("int", Load()), Constant(1), 1)
    ])
    assert "Members" not in parser.doc["test_module.E"]


# LLM-generated content at query #29
#--------------------------

```python
def test_is_public_evaluates_true():
    parser = Parser()
    parser.doc = {"module.name": "content"}
    parser.root = {"module.name": "module"}
    parser.imp = {"module": set()}
    assert parser.is_public("module.name")


# LLM-generated content at query #30
#--------------------------

```python
def test_is_enum_true_when_bases_contain_enum():
    parser = Parser()
    bases = ["enum.Enum", "other.Base"]
    parser.class_api("root", "name", bases, [])
    assert "Enums" in parser.doc["name"]

def test_is_enum_false_when_bases_do_not_contain_enum():
    parser = Parser()
    bases = ["other.Base"]
    parser.class_api("root", "name", bases, [])
    assert "Enums" not in parser.doc["name"]


# LLM-generated content at query #31
#--------------------------

```
def test_globals_ann_assign():
    parser = Parser()
    node = AnnAssign(target=Name(id="x", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=42))
    parser.globals("root", node)
    assert parser.alias["root.x"] == "42"
    assert parser.const["root.x"] == "int"

def test_globals_assign_with_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id="y", ctx=Store())], value=Constant(value="hello"), type_comment="str")
    parser.globals("root", node)
    assert parser.alias["root.y"] == "'hello'"
    assert parser.const["root.y"] == "str"

def test_globals_assign_without_type_comment():
    parser = Parser()
    node = Assign(targets=[Name(id="z", ctx=Store())], value=Constant(value=3.14))
    parser.globals("root", node)
    assert parser.alias["root.z"] == "3.14"
    assert parser.const["root.z"] == "float"

def test_globals_assign_tuple():
    parser = Parser()
    node = Assign(targets=[Name(id="__all__", ctx=Store())], value=Tuple(elts=[Constant(value="x"), Constant(value="y")], ctx=Load()))
    parser.globals("root", node)
    assert parser.imp["root"] == {"root.x", "root.y"}

def test_globals_assign_list():
    parser = Parser()
    node = Assign(targets=[Name(id="__all__", ctx=Store())], value=List(elts=[Constant(value="a"), Constant(value="b")], ctx=Load()))
    parser.globals("root", node)
    assert parser.imp["root"] == {"root.a", "root.b"}


# LLM-generated content at query #32
#--------------------------

```
def test_func_api_with_posonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[arg(arg='arg1', annotation=None)], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert 'arg1' in p.doc['name']

def test_func_api_with_args_and_defaults():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='arg1', annotation=None), arg(arg='arg2', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[None, Name(id='default', ctx=Load())])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert 'arg1' in p.doc['name'] and 'arg2' in p.doc['name']

def test_func_api_with_vararg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '*args' in p.doc['name']

def test_func_api_with_kwonlyargs():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[arg(arg='kwarg1', annotation=None)], kw_defaults=[None], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert 'kwarg1' in p.doc['name']

def test_func_api_with_kwarg():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=arg(arg='kwargs', annotation=None), defaults=[])
    p.func_api('root', 'name', args, None, has_self=False, cls_method=False)
    assert '**kwargs' in p.doc['name']

def test_func_api_with_return_annotation():
    p = Parser()
    args = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, Name(id='str', ctx=Load()), has_self=False, cls_method=False)
    assert 'return' in p.doc['name']

def test_func_api_with_self_param():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='self', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=False)
    assert 'self' in p.doc['name']

def test_func_api_with_cls_param():
    p = Parser()
    args = arguments(posonlyargs=[], args=[arg(arg='cls', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    p.func_api('root', 'name', args, None, has_self=True, cls_method=True)
    assert 'cls' in p.doc['name']


# LLM-generated content at query #33
#--------------------------

```
def test_is_public_returns_true_when_s_is_root_and_all_l_contains_s():
    p = Parser()
    p.imp = {'root': {'root'}}
    p.root = {'root': 'root'}
    p.doc = {}
    p.const = {}
    assert p.is_public('root') == True

def test_is_public_returns_true_when_parent_s_in_all_l():
    p = Parser()
    p.imp = {'root': {'parent'}}
    p.root = {'child': 'root'}
    p.doc = {}
    p.const = {}
    assert p.is_public('child') == True

def test_is_public_returns_true_when_s_in_all_l():
    p = Parser()
    p.imp = {'root': {'child'}}
    p.root = {'child': 'root'}
    p.doc = {}
    p.const = {}
    assert p.is_public('child') == True


# LLM-generated content at query #34
#--------------------------

```
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    p.globals('root', node)
    assert p.alias['root.x'] == '42'
    assert p.const['root.x'] == 'int'

def test_globals_with_assign():
    p = Parser()
    node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value='hello'))
    p.globals('root', node)
    assert p.alias['root.y'] == "'hello'"
    assert p.const['root.y'] == 'str'

def test_globals_with_assign_and_type_comment():
    p = Parser()
    node = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14), type_comment='float')
    p.globals('root', node)
    assert p.alias['root.z'] == '3.14'
    assert p.const['root.z'] == 'float'

def test_globals_with_assign_to_all():
    p = Parser()
    node = Assign(targets=[Name(id='__all__', ctx=Store())], value=List(elts=[Constant(value='x'), Constant(value='y')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.x', 'root.y'}

def test_globals_with_non_name_target():
    p = Parser()
    node = Assign(targets=[Subscript(value=Name(id='a', ctx=Load()), slice=Constant(value=0))], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const

def test_globals_with_multiple_targets():
    p = Parser()
    node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=1))
    p.globals('root', node)
    assert not p.alias
    assert not p.const

def test_globals_with_non_constant_value():
    p = Parser()
    node = Assign(targets=[Name(id='x', ctx=Store())], value=Name(id='y', ctx=Load()))
    p.globals('root', node)
    assert p.alias['root.x'] == 'y'
    assert p.const.get('root.x', ANY) == ANY

def test_globals_with_uppercase_name():
    p = Parser()
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=100))
    p.globals('root', node)
    assert p.alias['root.CONST'] == '100'
    assert p.const['root.CONST'] == 'int'

def test_globals_with_uppercase_name_and_existing_const():
    p = Parser()
    p.const['root.CONST'] = 'float'
    node = Assign(targets=[Name(id='CONST', ctx=Store())], value=Constant(value=100))
    p.globals('root', node)
    assert p.alias['root.CONST'] == '100'
    assert p.const['root.CONST'] == 'float'


# LLM-generated content at query #35
#--------------------------

```
def test_is_public_returns_false_when_no_children_with_public_family():
    p = Parser()
    p.imp = {'root': set()}
    p.doc = {'root': 'doc'}
    p.const = {'root': 'const'}
    p.root = {'root': 'root', 'root.child': 'root'}
    assert not p.is_public('root')


# LLM-generated content at query #36
#--------------------------

```python
def test_visit_Attribute_with_Name_value():
    node = Attribute(Name('typing', Load()), 'Any', Load())
    resolver = Resolver('root', {})
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'Any'


# LLM-generated content at query #37
#--------------------------

```
def test_const_type_with_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_constant_float():
    node = Constant(value=3.14)
    assert const_type(node) == "float"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_with_constant_bool():
    node = Constant(value=True)
    assert const_type(node) == "bool"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[], ctx=Load())
    assert const_type(node) == "tuple[]"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_mixed_tuple():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "tuple[Any]"

def test_const_type_with_empty_list():
    node = List(elts=[], ctx=Load())
    assert const_type(node) == "list[]"

def test_const_type_with_list_of_floats():
    node = List(elts=[Constant(value=1.1), Constant(value=2.2)], ctx=Load())
    assert const_type(node) == "list[float, float]"

def test_const_type_with_empty_set():
    node = Set(elts=[])
    assert const_type(node) == "set[]"

def test_const_type_with_set_of_strings():
    node = Set(elts=[Constant(value="x"), Constant(value="y")])
    assert const_type(node) == "set[str, str]"

def test_const_type_with_empty_dict():
    node = Dict(keys=[], values=[])
    assert const_type(node) == "dict[]"

def test_const_type_with_dict_of_int_to_str():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "dict[int, int][str, str]"

def test_const_type_with_builtin_func_call():
    node = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "int"

def test_const_type_with_unknown_node():
    node = Name(id="x", ctx=Load())
    assert const_type(node) == "Any"


# LLM-generated content at query #38
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    p = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='x', annotation=Name(id='int', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_self_no_cls_method():
    p = Parser()
    args = [arg(arg='self', annotation=None), arg(arg='x', annotation=Name(id='int', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_without_self():
    p = Parser()
    args = [arg(arg='x', annotation=Name(id='int', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int']

def test_func_ann_with_star_arg():
    p = Parser()
    args = [arg(arg='*', annotation=None), arg(arg='x', annotation=Name(id='int', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['', 'int']

def test_func_ann_with_no_annotation():
    p = Parser()
    args = [arg(arg='x', annotation=None)]
    result = list(p.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['typing.Any']

def test_func_ann_with_multiple_args():
    p = Parser()
    args = [
        arg(arg='self', annotation=None),
        arg(arg='x', annotation=Name(id='int', ctx=Load())),
        arg(arg='y', annotation=Name(id='str', ctx=Load())),
        arg(arg='z', annotation=None)
    ]
    result = list(p.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str', 'typing.Any']


# LLM-generated content at query #39
#--------------------------

```python
def test_const_type_dict():
    node = Dict(keys=[], values=[])
    assert const_type(node) == 'dict'

def test_const_type_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == 'tuple'

def test_const_type_list():
    node = List(elts=[])
    assert const_type(node) == 'list'

def test_const_type_set():
    node = Set(elts=[])
    assert const_type(node) == 'set'

def test_const_type_call_with_valid_func():
    node = Call(func=Name(id='int', ctx=Load()), args=[], keywords=[])
    assert const_type(node) == 'int'

def test_const_type_call_with_invalid_func():
    node = Call(func=Name(id='invalid_func', ctx=Load()), args=[], keywords=[])
    assert const_type(node) == ANY

def test_const_type_constant():
    node = Constant(value=42)
    assert const_type(node) == 'int'


# LLM-generated content at query #40
#--------------------------

```
def test_parser_default_constructor():
    p = Parser()
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is False
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_new_constructor():
    p = Parser.new(link=False, level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_post_init_with_toc():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True

def test_parser_post_init_without_toc():
    p = Parser(toc=False)
    assert p.link is True
    assert p.toc is False


# LLM-generated content at query #41
#--------------------------

```
def test_func_ann_yields_resolved_annotation_when_annotation_exists():
    parser = Parser()
    mock_arg = arg(arg='test_arg', annotation=Name(id='int', ctx=Load()))
    args = [mock_arg]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int']


# LLM-generated content at query #42
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_syntax_string():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_valid_string():
    resolver = Resolver("root", {})
    node = Constant("valid_syntax")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)


# LLM-generated content at query #43
#--------------------------

```python
def test__e_type_empty_sequence():
    assert _e_type() == ""

def test__e_type_empty_element():
    assert _e_type([]) == ""

def test__e_type_single_element_single_constant():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)]) == "[int]"

def test__e_type_multiple_elements_single_constant():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1)], [Constant(2)]) == "[int, int]"

def test__e_type_multiple_elements_multiple_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant(2)]) == "[int, int]"

def test__e_type_multiple_elements_different_types():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), Constant("a")]) == ""

def test__e_type_non_constant_element():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([1]) == ""

def test__e_type_mixed_constants_and_non_constants():
    class Constant:
        def __init__(self, value):
            self.value = value
    assert _e_type([Constant(1), 2]) == ""


