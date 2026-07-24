####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports_with_import_node():
    p = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname=None)])
    p.imports(root, node)
    assert p.alias["test.module.os"] == "os"

def test_imports_with_import_node_and_asname():
    p = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname="operating_system")])
    p.imports(root, node)
    assert p.alias["test.module.operating_system"] == "os"

def test_imports_with_import_from_node_and_level():
    p = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    p.imports(root, node)
    assert p.alias["test.module.submodule.path"] == "test.module.os.path"

def test_imports_with_import_from_node_and_asname():
    p = Parser()
    root = "test.module"
    node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    p.imports(root, node)
    assert p.alias["test.module.dd"] == "collections.defaultdict"


# LLM-generated content at query #2
#--------------------------

```python
def test_load_docstring():
    p = Parser()
    p.doc = {'pkg.sub': '...', 'pkg.sub.func': '...'}
    p.docstring = {}
    m = ModuleType('pkg.sub')
    m.func = lambda: None
    m.func.__doc__ = "Function docstring"
    p.load_docstring('pkg.sub', m)
    assert p.docstring == {'pkg.sub.func': "```python\nFunction docstring\n```"}


# LLM-generated content at query #3
#--------------------------

```python
def test__defaults_empty_sequence():
    assert list(_defaults([])) == [" "]

def test__defaults_none_values():
    assert list(_defaults([None, None])) == [" ", " "]

def test__defaults_with_values_no_ampersand():
    assert list(_defaults([expr.Name(id='x'), expr.Constant(value=1)])) == ["`x`", "`1`"]

def test__defaults_with_values_with_ampersand():
    assert list(_defaults([expr.Constant(value='a&b')])) == ["<code>a&#38;b</code>"]

def test__defaults_with_values_with_pipe():
    assert list(_defaults([expr.Constant(value='a|b')])) == ["`<code>a&#124;b</code>`"]

def test__defaults_mixed_values():
    assert list(_defaults([None, expr.Constant(value='x&y'), expr.Name(id='z')])) == [" ", "<code>x&#38;y</code>", "`z`"]


# LLM-generated content at query #4
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_string():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_valid_name():
    resolver = Resolver("root", {"root.name": "alias"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"


# LLM-generated content at query #5
#--------------------------

```python
def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'__init__': ''}
    parser.const = {}
    parser.root = {'__init__': 'root'}
    assert parser.is_public('__init__') == True

def test_is_public_with_private_name_not_in_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {}
    parser.const = {}
    parser.root = {'_private': 'root'}
    assert parser.is_public('_private') == False

def test_is_public_with_public_name_in_all():
    parser = Parser()
    parser.imp = {'root': {'public'}}
    parser.doc = {}
    parser.const = {}
    parser.root = {'public': 'root'}
    assert parser.is_public('public') == True

def test_is_public_with_child_in_all():
    parser = Parser()
    parser.imp = {'root': {'parent.child'}}
    parser.doc = {'parent.child': ''}
    parser.const = {}
    parser.root = {'parent.child': 'root'}
    assert parser.is_public('parent.child') == True

def test_is_public_with_parent_in_all():
    parser = Parser()
    parser.imp = {'root': {'parent'}}
    parser.doc = {'parent.child': ''}
    parser.const = {}
    parser.root = {'parent.child': 'root'}
    assert parser.is_public('parent.child') == True

def test_is_public_with_const_not_in_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {}
    parser.const = {'CONST': 'int'}
    parser.root = {'CONST': 'root'}
    assert parser.is_public('CONST') == True


# LLM-generated content at query #6
#--------------------------

```python
def test_load_docstring_skips_non_matching_names():
    p = Parser()
    p.doc = {'module.submodule': '', 'other.module': ''}
    p.load_docstring('module', type('m', (), {'submodule': ''}))
    assert 'module.submodule' in p.docstring
    assert 'other.module' not in p.docstring


# LLM-generated content at query #7
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Assign(targets=[Name(id="attr2", ctx=Store())], value=Constant(value=42)),
        Delete(targets=[Name(id="attr3", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "attr3" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="OPTION1", ctx=Store()), annotation=None, value=None),
        AnnAssign(target=Name(id="OPTION2", ctx=Store()), annotation=None, value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "OPTION1" in parser.doc[name]
    assert "OPTION2" in parser.doc[name]

def test_class_api_with_no_bases_or_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #8
#--------------------------

```python
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is True  # toc overrides link to True in __post_init__
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_is_public_with_child_in_doc():
    p = Parser()
    p.imp = {'pkg': {'subpkg'}}
    p.doc = {'pkg.subpkg': '', 'pkg.subpkg.module': ''}
    p.root = {'pkg.subpkg': 'pkg', 'pkg.subpkg.module': 'pkg.subpkg'}
    assert p.is_public('pkg.subpkg')


# LLM-generated content at query #10
#--------------------------

```python
def test_attr_simple_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.simple = "value"
    assert _attr(obj, "simple") == "value"

def test_attr_nested_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.nested = TestClass()
    obj.nested.attr = "nested_value"
    assert _attr(obj, "nested.attr") == "nested_value"

def test_attr_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "nonexistent") is None

def test_attr_nonexistent_nested_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.nested = TestClass()
    assert _attr(obj, "nested.nonexistent") is None

def test_attr_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "") is None

def test_attr_multiple_levels():
    class TestClass:
        pass
    obj = TestClass()
    obj.level1 = TestClass()
    obj.level1.level2 = TestClass()
    obj.level1.level2.level3 = "deep_value"
    assert _attr(obj, "level1.level2.level3") == "deep_value"

def test_attr_with_none_intermediate():
    class TestClass:
        pass
    obj = TestClass()
    obj.level1 = None
    assert _attr(obj, "level1.level2") is None


# LLM-generated content at query #11
#--------------------------

```python
def test_walk_body_empty_sequence():
    assert list(walk_body([])) == []

def test_walk_body_single_statement():
    stmt = Pass()
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_multiple_statements():
    stmt1 = Pass()
    stmt2 = Pass()
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

def test_walk_body_if_statement():
    if_node = If(test=Name(id='x'), body=[Pass()], orelse=[Pass()])
    assert list(walk_body([if_node])) == [Pass(), Pass()]

def test_walk_body_nested_if_statements():
    inner_if = If(test=Name(id='y'), body=[Pass()], orelse=[Pass()])
    outer_if = If(test=Name(id='x'), body=[inner_if], orelse=[Pass()])
    assert list(walk_body([outer_if])) == [Pass(), Pass(), Pass()]

def test_walk_body_try_statement():
    try_node = Try(
        body=[Pass()],
        handlers=[ExceptHandler(body=[Pass()])],
        orelse=[Pass()],
        finalbody=[Pass()]
    )
    assert list(walk_body([try_node])) == [Pass(), Pass(), Pass(), Pass()]

def test_walk_body_try_with_multiple_handlers():
    try_node = Try(
        body=[Pass()],
        handlers=[ExceptHandler(body=[Pass()]), ExceptHandler(body=[Pass()])],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [Pass(), Pass(), Pass()]

def test_walk_body_mixed_statements():
    stmt1 = Pass()
    if_node = If(test=Name(id='x'), body=[Pass()], orelse=[Pass()])
    try_node = Try(body=[Pass()], handlers=[], orelse=[], finalbody=[])
    assert list(walk_body([stmt1, if_node, try_node])) == [stmt1, Pass(), Pass(), Pass()]


# LLM-generated content at query #12
#--------------------------

```python
def test_is_public_with_all_listed():
    parser = Parser()
    parser.imp = {
        'root': {'root.sub1', 'root.sub2'},
        'root.sub1': {'root.sub1.item1', 'root.sub1.item2'},
    }
    parser.doc = {
        'root': '',
        'root.sub1': '',
        'root.sub1.item1': '',
        'root.sub2': '',
    }
    parser.root = {
        'root': 'root',
        'root.sub1': 'root',
        'root.sub1.item1': 'root.sub1',
        'root.sub2': 'root',
    }
    assert parser.is_public('root.sub1')
    assert parser.is_public('root.sub1.item1')
    assert not parser.is_public('root.sub1.item2')
    assert parser.is_public('root.sub2')
    assert not parser.is_public('root.sub3')

def test_is_public_without_all_listed():
    parser = Parser()
    parser.imp = {
        'root': set(),
    }
    parser.doc = {
        'root': '',
        'root.public': '',
        'root._private': '',
    }
    parser.root = {
        'root': 'root',
        'root.public': 'root',
        'root._private': 'root',
    }
    assert parser.is_public('root.public')
    assert not parser.is_public('root._private')


# LLM-generated content at query #13
#--------------------------

```python
def test_class_api_with_empty_bases():
    parser = Parser()
    parser.class_api("root", "name", [], [])
    assert "Bases" not in parser.doc["name"]


# LLM-generated content at query #14
#--------------------------

```python
def test__attr_returns_none_when_intermediate_attribute_is_none():
    class MockObject:
        pass

    obj = MockObject()
    obj.a = None
    assert _attr(obj, 'a.b') is None


# LLM-generated content at query #15
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "| `BaseClass` |" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="MEMBER1", ctx=Store()), annotation=Constant(value=1), simple=1),
        AnnAssign(target=Name(id="MEMBER2", ctx=Store()), annotation=Constant(value=2), simple=1)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "| `MEMBER1` | `MEMBER2` |" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), simple=1),
        Assign(targets=[Name(id="another_attr", ctx=Store())], value=Constant(value=42))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "| `public_attr` | `int` |" in parser.doc[name]
    assert "| `another_attr` | `int` |" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), simple=1),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="_private_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), simple=1)
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #16
#--------------------------

```python
def test_class_api_with_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=Constant(value=1)
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]


# LLM-generated content at query #17
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = [Name(id="Base1"), Name(id="Base2")]
    body = []
    parser.class_api(root, name, bases, body)
    expected = "| Bases |\n|:---:|\n| `Base1` | `Base2` |\n\n"
    assert parser.doc[name] == expected

def test_class_api_with_enum():
    parser = Parser()
    root = "module"
    name = "module.EnumClass"
    bases = [Name(id="enum.Enum")]
    body = [AnnAssign(target=Name(id="A"), annotation=Name(id="int")), AnnAssign(target=Name(id="B"), annotation=Name(id="int"))]
    parser.class_api(root, name, bases, body)
    expected = "| Enums |\n|:---:|\n| A |\n| B |\n\n"
    assert parser.doc[name] == expected

def test_class_api_with_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [AnnAssign(target=Name(id="attr1"), annotation=Name(id="int")), AnnAssign(target=Name(id="attr2"), annotation=Name(id="str"))]
    parser.class_api(root, name, bases, body)
    expected = "| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n"
    assert parser.doc[name] == expected

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [AnnAssign(target=Name(id="attr1"), annotation=Name(id="int")), Delete(targets=[Name(id="attr1")])]
    parser.class_api(root, name, bases, body)
    expected = ""
    assert parser.doc[name] == expected

def test_class_api_with_assign_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [Assign(targets=[Name(id="attr1")], value=Constant(value=1))]
    parser.class_api(root, name, bases, body)
    expected = "| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n\n"
    assert parser.doc[name] == expected


# LLM-generated content at query #18
#--------------------------

```python
def test__e_type_with_empty_input():
    assert _e_type() == ""

def test__e_type_with_none_element():
    assert _e_type([None]) == ""

def test__e_type_with_non_constant_element():
    assert _e_type([1, "a"]) == ""

def test__e_type_with_single_constant_element():
    assert _e_type([1]) == "[int]"

def test__e_type_with_multiple_same_type_constant_elements():
    assert _e_type([1, 2, 3]) == "[int]"

def test__e_type_with_multiple_different_type_constant_elements():
    assert _e_type([1, "a"]) == "[Any]"

def test__e_type_with_multiple_sequences():
    assert _e_type([1, 2], ["a", "b"]) == "[int, str]"

def test__e_type_with_mixed_sequences():
    assert _e_type([1, "a"], [2, 3]) == "[Any, int]"


# LLM-generated content at query #19
#--------------------------

```python
def test_parse_basic_module():
    p = Parser()
    script = """
\"\"\"Module docstring.\"\"\"
x = 1
def foo():
    \"\"\"Function docstring.\"\"\"
    pass
"""
    p.parse("test_module", script)
    assert p.doc["test_module"] == "### Module `{}`\n<a id=\"{}\"></a>\n\n"
    assert p.level["test_module"] == 0
    assert p.imp["test_module"] == set()
    assert p.root["test_module"] == "test_module"
    assert p.docstring["test_module"] == "```python\n\"\"\"Module docstring.\"\"\"\n```"
    assert p.doc["test_module.foo"] == "#### foo()\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n"
    assert p.docstring["test_module.foo"] == "```python\n    \"\"\"Function docstring.\"\"\"\n```"

def test_parse_with_imports():
    p = Parser()
    script = """
import os
from sys import path
x = 1
"""
    p.parse("test_module", script)
    assert p.alias["test_module.os"] == "os"
    assert p.alias["test_module.path"] == "sys.path"

def test_parse_with_globals():
    p = Parser()
    script = """
X = 1
Y: int = 2
"""
    p.parse("test_module", script)
    assert p.alias["test_module.X"] == "1"
    assert p.alias["test_module.Y"] == "2"
    assert p.const["test_module.X"] == "int"
    assert p.const["test_module.Y"] == "int"

def test_parse_with_all():
    p = Parser()
    script = """
__all__ = ['foo', 'bar']
def foo():
    pass
def bar():
    pass
"""
    p.parse("test_module", script)
    assert p.imp["test_module"] == {"test_module.foo", "test_module.bar"}

def test_parse_with_class():
    p = Parser()
    script = """
class Foo:
    \"\"\"Class docstring.\"\"\"
    def bar(self):
        pass
"""
    p.parse("test_module", script)
    assert p.doc["test_module.Foo"] == "#### class Foo\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n"
    assert p.docstring["test_module.Foo"] == "```python\n    \"\"\"Class docstring.\"\"\"\n```"
    assert p.doc["test_module.Foo.bar"] == "##### bar()\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n"

def test_parse_with_decorators():
    p = Parser()
    script = """
@decorator
def foo():
    pass
"""
    p.parse("test_module", script)
    assert p.doc["test_module.foo"] == "#### foo()\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n| Decorators |\n|------------|\n|`@decorator`|"


# LLM-generated content at query #20
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=5),
        simple=1
    )
    parser.globals("module", node)
    assert parser.alias["module.x"] == "5"
    assert parser.const["module.x"] == "int"
    assert parser.root["module.x"] == "module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value="hello")
    )
    parser.globals("module", node)
    assert parser.alias["module.y"] == "'hello'"
    assert parser.const["module.y"] == "str"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="foo"), Constant(value="bar")])
    )
    parser.globals("module", node)
    assert parser.imp["module"] == {"module.foo", "module.bar"}

def test_globals_with_non_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=BinOp(left=Constant(value=1), op=Add(), right=Constant(value=2))
    )
    parser.globals("module", node)
    assert parser.alias["module.z"] == "1 + 2"
    assert parser.const["module.z"] == "Any"


# LLM-generated content at query #21
#--------------------------

```python
def test_class_api_with_enum():
    parser = Parser()
    parser.parse('test', 'from enum import Enum\nclass Test(Enum):\n    A = 1\n    B = 2')
    assert 'Enums' in parser.doc['test.Test']


# LLM-generated content at query #22
#--------------------------

```python
def test_imports_predicate_false():
    p = Parser()
    node = ImportFrom(module=None, names=[], level=0)
    p.imports("root", node)
    assert len(p.alias) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_line_31():
    p = Parser()
    p.class_api("root", "name", [], [Delete(targets=[Subscript()])])
    assert len(p.doc["name"]) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_load_docstring_updates_docstring_when_doc_is_not_none():
    p = Parser()
    p.doc = {'module.submodule.func': ''}
    p.docstring = {}
    m = type('MockModule', (), {'submodule': type('MockSubmodule', (), {'func': lambda: None})})()
    m.submodule.func.__doc__ = "This is a test docstring."
    p.load_docstring('module', m)
    assert 'module.submodule.func' in p.docstring
    assert p.docstring['module.submodule.func'] == "This is a test docstring."


# LLM-generated content at query #25
#--------------------------

```python
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is False
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_post_init_toc_true():
    p = Parser(toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #26
#--------------------------

```python
def test_func_ann_with_self_and_annotation():
    parser = Parser()
    args = [arg('self', Name('Self', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method_and_annotation():
    parser = Parser()
    args = [arg('cls', Name('type[Self]', Load())), arg('x', Name('int', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_without_annotation():
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('x', Name('int', Load())), arg('*', None), arg('y', Name('str', Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']

def test_func_ann_with_self_ty_resolution():
    parser = Parser()
    parser.alias['root.Self'] = 'root.Parent'
    args = [arg('self', Name('Self', Load())), arg('x', Name('Self', Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False, self_ty='root.Parent'))
    assert result == ['Self', 'root.Parent']


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_true():
    e1 = Constant(1)
    e2 = Constant(2)
    assert _e_type([e1, e2]) == "[Int]"


# LLM-generated content at query #28
#--------------------------

```python
def test_func_ann_yields_type_self_when_cls_method_is_true():
    parser = Parser()
    args = [arg('self', None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'


# LLM-generated content at query #29
#--------------------------

```python
def test_globals_type_comment_not_none():
    p = Parser()
    node = Assign(targets=[Name(id='x')], value=Constant(value=1), type_comment='int')
    p.globals('root', node)
    assert p.alias['root.x'] == '1'
    assert p.const.get('root.x') is None


# LLM-generated content at query #30
#--------------------------

```python
def test_func_api_with_no_args_and_no_return():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | return |\n|:---:|:---:|\n|  | ANY |\n\n"

def test_func_api_with_args_and_return():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("x", None), arg("y", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name(id="int", ctx=Load())
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| x | y | int |\n\n"

def test_func_api_with_defaults():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("x", None), arg("y", None)], kwonlyargs=[], kw_defaults=[], defaults=[Constant(value=1), Constant(value=2)])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| x | y | ANY |\n| 1 | 2 |  |\n\n"

def test_func_api_with_self():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("self", None), arg("x", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=True, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| Self | x | ANY |\n\n"

def test_func_api_with_cls_method():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("cls", None), arg("x", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=True, cls_method=True)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| type[Self] | x | ANY |\n\n"

def test_func_api_with_varargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("x", None)], kwonlyargs=[], kw_defaults=[], defaults=[], vararg=arg("args", None))
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| x | *args | ANY |\n\n"

def test_func_api_with_kwargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("x", None)], kwonlyargs=[], kw_defaults=[], defaults=[], kwarg=arg("kwargs", None))
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| x | **kwargs | ANY |\n\n"

def test_func_api_with_posonlyargs():
    parser = Parser()
    args = arguments(posonlyargs=[arg("x", None)], args=[arg("y", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | arg | return |\n|:---:|:---:|:---:|:---:|\n| x | / | y | ANY |\n\n"

def test_func_api_with_kwonlyargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[arg("x", None)], kw_defaults=[Constant(value=1)], defaults=[])
    returns = None
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | return |\n|:---:|:---:|\n| x | ANY |\n| 1 |  |\n\n"

def test_func_api_with_annotations():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg("x", Name(id="int", ctx=Load())), arg("y", Name(id="str", ctx=Load()))], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name(id="bool", ctx=Load())
    parser.func_api("root", "func", args, returns, has_self=False, cls_method=False)
    assert parser.doc["func"] == "# func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| arg | arg | return |\n|:---:|:---:|:---:|\n| int | str | bool |\n\n"


# LLM-generated content at query #31
#--------------------------

```python
def test_const_type_with_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_list_of_mixed_types():
    node = List(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "list[Any]"

def test_const_type_with_dict_of_same_types():
    keys = [Constant(value="a"), Constant(value="b")]
    values = [Constant(value=1), Constant(value=2)]
    node = Dict(keys=keys, values=values)
    assert const_type(node) == "dict[str, int]"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == "tuple[]"

def test_const_type_with_call_to_bool():
    node = Call(func=Name(id="bool"))
    assert const_type(node) == "bool"

def test_const_type_with_call_to_list():
    node = Call(func=Name(id="list"))
    assert const_type(node) == "list"

def test_const_type_with_unsupported_node():
    node = BinOp()
    assert const_type(node) == "Any"


# LLM-generated content at query #32
#--------------------------

```python
def test_func_ann_yields_empty_string_for_star_arg():
    parser = Parser()
    args = [arg('*', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == [""]


# LLM-generated content at query #33
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Attribute_non_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="other", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "List"


# LLM-generated content at query #34
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    args = arguments(posonlyargs=[arg('x', None), arg('y', None)], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=False, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| x | y | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n"

def test_func_api_with_keyword_args():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[arg('x', None), arg('y', None)], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=False, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| * | x | y | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n"

def test_func_api_with_varargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=arg('args', None))
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=False, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"

def test_func_api_with_kwargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=False, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"

def test_func_api_with_defaults():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('x', None)], defaults=[Constant(1)], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=False, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| x | return |\n|:---:|:---:|\n| `Any` | `Any` |\n| `1` |  |\n\n"

def test_func_api_with_self():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('self', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=True, cls_method=False)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| Self | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"

def test_func_api_with_cls_method():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('cls', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'func', args, returns, has_self=True, cls_method=True)
    assert parser.doc['func'] == '#' * (parser.b_level + 2) + " func()\n\n*Full name:* `func`\n<a id=\"func\"></a>\n\n| type[Self] | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"


# LLM-generated content at query #35
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    root = "test_module"
    node = Assign(targets=[Name(id="test_var")], value=Constant(value=42))
    parser.globals(root, node)
    assert parser.imp[root] == set()


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_19_evaluates_to_false():
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name(id="x"), Name(id="y")],
            value=Constant(value=1)
        )
    ]
    parser.class_api(root, name, bases, body)
    assert len(parser.doc[name]) == 0


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parser = Parser()
    node = Assign(targets=[Name(id='test')], value=Constant(value=42))
    parser.globals('root', node)
    assert 'root.test' not in parser.imp


# LLM-generated content at query #38
#--------------------------

```python
def test_func_api_no_args_no_return():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n|  |  |\n\n"
    assert parser.doc["root.test"] == expected

def test_func_api_with_args_and_return():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[arg("x", Constant(1)), arg("y", Constant(2))], kwonlyargs=[], kw_defaults=[], defaults=[Constant(1), Constant(2)]), returns=Constant(3), body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=False, cls_method=False)
    expected = "| arg | arg | return |\n|:---:|:---:|:---:|\n| `x` | `y` | `3` |\n| `1` | `2` |  |\n\n"
    assert parser.doc["root.test"] == expected

def test_func_api_with_self():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[arg("self", Name("Self", Load()))], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=True, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `Self` |  |\n\n"
    assert parser.doc["root.test"] == expected

def test_func_api_with_cls_method():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[arg("cls", Name("type[Self]", Load()))], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=True, cls_method=True)
    expected = "| arg | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n"
    assert parser.doc["root.test"] == expected

def test_func_api_with_varargs():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[], vararg=arg("args", None), kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `*args` |  |\n\n"
    assert parser.doc["root.test"] == expected

def test_func_api_with_kwargs():
    parser = Parser()
    func_def = FunctionDef(name="test", args=arguments(posonlyargs=[], args=[], kwarg=arg("kwargs", None), kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, body=[])
    parser.func_api("root", "root.test", func_def.args, func_def.returns, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `**kwargs` |  |\n\n"
    assert parser.doc["root.test"] == expected


# LLM-generated content at query #39
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #40
#--------------------------

```python
def test_imports_with_asname():
    p = Parser()
    node = ImportFrom(module="sys", names=[alias(name="path", asname="sp")], level=0)
    p.imports("test", node)
    assert p.alias.get("test.sp") == "sys.path"


# LLM-generated content at query #41
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=123),
        type_comment=None
    )
    parser.globals(root, node)
    assert "__all__" not in parser.imp[root]


# LLM-generated content at query #42
#--------------------------

```python
def test_walk_body_handlers_empty():
    body = [Try(body=[], handlers=[], orelse=[], finalbody=[])]
    assert list(walk_body(body)) == [Try(body=[], handlers=[], orelse=[], finalbody=[])]


# LLM-generated content at query #43
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = [Name(id="Base1"), Name(id="Base2")]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "| Base1 | Base2 |" in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyEnum"
    bases = [Name(id="enum.Enum")]
    body = [
        AnnAssign(target=Name(id="A"), annotation=Name(id="int"), value=Constant(value=1)),
        AnnAssign(target=Name(id="B"), annotation=Name(id="int"), value=Constant(value=2))
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "| A | B |" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr"), annotation=Name(id="int"), value=Constant(value=1)),
        Assign(targets=[Name(id="another_attr")], value=Constant(value="value"), type_comment="str")
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "| public_attr | another_attr |" in parser.doc[name]
    assert "| int | str |" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="attr1"), annotation=Name(id="int"), value=Constant(value=1)),
        Delete(targets=[Name(id="attr1")])
    ]
    parser.class_api(root, name, bases, body)
    assert "attr1" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "test.module"
    name = "test.module.MyClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="_private_attr"), annotation=Name(id="int"), value=Constant(value=1))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_evaluates_to_false():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='foo')]))
    p.globals('root', node)
    assert '__all__' not in p.imp['root']


# LLM-generated content at query #45
#--------------------------

```python
def test_isinstance_call_and_name_or_attribute():
    node = Call(func=Name(id='test'), args=[])
    assert isinstance(node, Call) and isinstance(node.func, (Name, Attribute))


# LLM-generated content at query #46
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver(root="root", alias={}, self_ty="T")
    node = Name("T", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver(root="root", alias={"root.name": "alias.value"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "value"

def test_visit_Name_without_alias():
    resolver = Resolver(root="root", alias={})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver(root="root", alias={"root.name": "TypeVar('T')", "root.TypeVar": "typing.TypeVar"})
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #47
#--------------------------

```python
def test_func_api_with_no_args():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| return |\n|:-----:|\n|  |\n\n"

def test_func_api_with_args_and_no_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg("a", None), arg("b", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| a | b | return |\n|:---:|:---:|:-----:|\n| ANY | ANY |  |\n\n"

def test_func_api_with_args_and_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg("a", None), arg("b", None)], kwonlyargs=[], kw_defaults=[], defaults=[Constant(value=1), Constant(value=2)])
    returns = None
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| a | b | return |\n|:---:|:---:|:-----:|\n| ANY | ANY |  |\n| `1` | `2` |  |\n\n"

def test_func_api_with_self_and_cls_method():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg("self", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "name", node, returns, has_self=True, cls_method=True)
    assert parser.doc["name"] == "| self | return |\n|:-----:|:-----:|\n| type[Self] |  |\n\n"

def test_func_api_with_self_and_not_cls_method():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg("self", None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    parser.func_api("root", "name", node, returns, has_self=True, cls_method=False)
    assert parser.doc["name"] == "| self | return |\n|:-----:|:-----:|\n| Self |  |\n\n"

def test_func_api_with_vararg():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[], vararg=arg("args", None))
    returns = None
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| *args | return |\n|:------:|:-----:|\n|  |  |\n\n"

def test_func_api_with_kwarg():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[], kwarg=arg("kwargs", None))
    returns = None
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| **kwargs | return |\n|:---------:|:-----:|\n|  |  |\n\n"

def test_func_api_with_returns():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = Name(id="int", ctx=Load())
    parser.func_api("root", "name", node, returns, has_self=False, cls_method=False)
    assert parser.doc["name"] == "| return |\n|:-----:|\n| `int` |\n\n"


# LLM-generated content at query #48
#--------------------------

```python
def test_globals_ann_assign_with_value():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x"),
        annotation=Name(id="int"),
        value=Constant(value=1),
        simple=1
    )
    parser.globals("root", node)
    assert parser.alias["root.x"] == "1"
    assert parser.const["root.x"] == "int"
    assert parser.root["root.x"] == "root"

def test_globals_ann_assign_without_value():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x"),
        annotation=Name(id="int"),
        value=None,
        simple=1
    )
    parser.globals("root", node)
    assert "root.x" not in parser.alias
    assert "root.x" not in parser.const
    assert "root.x" not in parser.root

def test_globals_assign_with_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="x")],
        value=Constant(value=1),
        type_comment="int"
    )
    parser.globals("root", node)
    assert parser.alias["root.x"] == "1"
    assert parser.const["root.x"] == "int"
    assert parser.root["root.x"] == "root"

def test_globals_assign_without_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="x")],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals("root", node)
    assert parser.alias["root.x"] == "1"
    assert parser.const["root.x"] == "int"
    assert parser.root["root.x"] == "root"

def test_globals_assign_multiple_targets():
    parser = Parser()
    node = Assign(
        targets=[Name(id="x"), Name(id="y")],
        value=Constant(value=1),
        type_comment=None
    )
    parser.globals("root", node)
    assert "root.x" not in parser.alias
    assert "root.x" not in parser.const
    assert "root.x" not in parser.root

def test_globals_all_filter():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="x"), Constant(value="y")]),
        type_comment=None
    )
    parser.globals("root", node)
    assert parser.imp["root"] == {"root.x", "root.y"}

def test_globals_all_filter_non_string():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value=1), Constant(value=2)]),
        type_comment=None
    )
    parser.globals("root", node)
    assert parser.imp["root"] == set()


# LLM-generated content at query #49
#--------------------------

```python
def test_func_api_with_posonlyargs_and_defaults():
    parser = Parser()
    node = arguments(
        posonlyargs=[arg('a', None), arg('b', None)],
        args=[arg('c', None)],
        defaults=[Constant(value=1)],
        kwonlyargs=[arg('d', None)],
        kw_defaults=[Constant(value=2)],
        kwarg=arg('kwargs', None)
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        '### func()\n\n'
        '*Full name:* `root.func`\n'
        '<a id="root-func"></a>\n\n'
        '| a | b | / | c | * | d | **kwargs | return |\n'
        '|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n'
        '| `Any` | `Any` |  | `Any` |  | `Any` | `Any` | `Any` |\n'
        '| `Any` | `Any` |  | 1 |  | 2 |  |  |\n'
    )

def test_func_api_with_vararg_and_returns():
    parser = Parser()
    node = arguments(
        args=[arg('a', None)],
        vararg=arg('args', None),
        kwarg=arg('kwargs', None)
    )
    returns = Name(id='int', ctx=Load())
    parser.func_api('root', 'func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        '### func()\n\n'
        '*Full name:* `root.func`\n'
        '<a id="root-func"></a>\n\n'
        '| a | *args | **kwargs | return |\n'
        '|:---:|:---:|:---:|:---:|\n'
        '| `Any` | `Any` | `Any` | `int` |\n'
    )

def test_func_api_with_has_self_and_cls_method():
    parser = Parser()
    node = arguments(
        args=[arg('self', None), arg('a', None)],
        defaults=[Constant(value=1)]
    )
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    assert parser.doc['root.func'] == (
        '### func()\n\n'
        '*Full name:* `root.func`\n'
        '<a id="root-func"></a>\n\n'
        '| self | a | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `type[Self]` | `Any` | `Any` |\n'
        '|  | 1 |  |\n'
    )

def test_func_api_with_annotations():
    parser = Parser()
    node = arguments(
        args=[arg('a', Name(id='int', ctx=Load())), arg('b', Name(id='str', ctx=Load()))],
        defaults=[Constant(value=1)]
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        '### func()\n\n'
        '*Full name:* `root.func`\n'
        '<a id="root-func"></a>\n\n'
        '| a | b | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `int` | `str` | `Any` |\n'
        '|  | 1 |  |\n'
    )


# LLM-generated content at query #50
#--------------------------

```python
def test_parser_constructor_default_values():
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

def test_parser_constructor_with_custom_values():
    p = Parser(link=False, b_level=2, toc=True)
    assert p.link is False
    assert p.b_level == 2
    assert p.toc is True

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is False
    assert p.b_level == 3
    assert p.toc is True

def test_parser_post_init_toc_true_sets_link_true():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True


# LLM-generated content at query #51
#--------------------------

```python
def test_func_ann_with_has_self_and_cls_method():
    p = Parser()
    args = [
        arg('self', None),
        arg('x', Name(id='int', ctx=Load())),
        arg('return', None)
    ]
    result = list(p.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', '']

def test_func_ann_with_has_self_no_cls_method():
    p = Parser()
    args = [
        arg('self', None),
        arg('x', Name(id='int', ctx=Load())),
        arg('return', None)
    ]
    result = list(p.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int', '']

def test_func_ann_without_has_self():
    p = Parser()
    args = [
        arg('x', Name(id='int', ctx=Load())),
        arg('return', None)
    ]
    result = list(p.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '']

def test_func_ann_with_star_arg():
    p = Parser()
    args = [
        arg('x', Name(id='int', ctx=Load())),
        arg('*', None),
        arg('return', None)
    ]
    result = list(p.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', '']

def test_func_ann_with_no_annotation():
    p = Parser()
    args = [
        arg('x', None),
        arg('return', None)
    ]
    result = list(p.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['Any', '']


# LLM-generated content at query #52
#--------------------------

```python
def test_globals_with_annassign():
    p = Parser()
    node = AnnAssign(
        target=Name(id='x', ctx=Store()),
        annotation=Name(id='int', ctx=Load()),
        value=Constant(value=5),
        simple=1
    )
    p.globals('root', node)
    assert p.alias['root.x'] == '5'
    assert p.const['root.x'] == 'int'
    assert p.root['root.x'] == 'root'

def test_globals_with_assign():
    p = Parser()
    node = Assign(
        targets=[Name(id='y', ctx=Store())],
        value=Constant(value='hello')
    )
    p.globals('root', node)
    assert p.alias['root.y'] == "'hello'"
    assert p.const['root.y'] == 'str'
    assert p.root['root.y'] == 'root'

def test_globals_with_all():
    p = Parser()
    node = Assign(
        targets=[Name(id='__all__', ctx=Store())],
        value=List(elts=[Constant(value='foo'), Constant(value='bar')])
    )
    p.globals('root', node)
    assert p.imp['root'] == {'root.foo', 'root.bar'}

def test_globals_ignores_complex_assign():
    p = Parser()
    node = Assign(
        targets=[Name(id='x', ctx=Store()), Name(id='y', ctx=Store())],
        value=Constant(value=1)
    )
    p.globals('root', node)
    assert 'root.x' not in p.alias
    assert 'root.y' not in p.alias

def test_globals_ignores_non_constant_all():
    p = Parser()
    node = Assign(
        targets=[Name(id='__all__', ctx=Store())],
        value=Name(id='some_var', ctx=Load())
    )
    p.globals('root', node)
    assert p.imp['root'] == set()


# LLM-generated content at query #53
#--------------------------

```python
def test_func_ann_with_annotation_and_cls_method():
    p = Parser()
    args = [arg(arg='self', annotation=Name(id='Self', ctx=Load()))]
    result = list(p.func_ann('root', args, has_self=True, cls_method=True))
    assert result[0] == 'type[Self]'


# LLM-generated content at query #54
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id="x"), Name(id="y")], value=Constant(value=42))
    parser.globals("root", node)
    assert "root.x" not in parser.alias
    assert "root.y" not in parser.alias


# LLM-generated content at query #55
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_custom():
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

def test_parser_constructor_post_init_toc():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True


# LLM-generated content at query #56
#--------------------------

```python
def test_func_api_with_posonlyargs():
    p = Parser()
    root = "test_module"
    name = "test_module.test_func"
    node = arguments(
        posonlyargs=[arg(arg="a", annotation=None), arg(arg="b", annotation=None)],
        args=[arg(arg="c", annotation=None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        vararg=None,
        kwarg=None
    )
    returns = None
    has_self = False
    cls_method = False
    p.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert p.doc[name] == "#" * (p.b_level + 2) + " test_func()\n\n*Full name:* `{}`\n<a id=\"{}\"></a>\n\n"


# LLM-generated content at query #57
#--------------------------

```python
def test_isinstance_d_not_name():
    p = Parser()
    p.parse('test', 'class Test:\n    del x[0]')
    assert isinstance(p.doc['test.Test'], str)


# LLM-generated content at query #58
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver(root="test", alias={}, self_ty="T")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #59
#--------------------------

```python
def test_globals_with_annassign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="test_var", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_assign_no_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == "test_module"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="public_func")])
    )
    parser.globals("test_module", node)
    assert parser.imp["test_module"] == {"test_module.public_func"}

def test_globals_with_non_uppercase():
    parser = Parser()
    node = Assign(
        targets=[Name(id="non_upper", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals("test_module", node)
    assert parser.alias["test_module.non_upper"] == "42"
    assert "test_module.non_upper" not in parser.const
    assert "test_module.non_upper" not in parser.root


# LLM-generated content at query #60
#--------------------------

```python
def test_parser_default_constructor():
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link == False
    assert p.b_level == 3
    assert p.toc == True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}

def test_parser_post_init_with_toc():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link == True
    assert p.b_level == 1
    assert p.toc == True


# LLM-generated content at query #61
#--------------------------

```python
def test_imports_with_import_node():
    parser = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, node)
    assert parser.alias["test.module.os"] == "os"

def test_imports_with_import_node_and_asname():
    parser = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, node)
    assert parser.alias["test.module.operating_system"] == "os"

def test_imports_with_import_from_node():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="os.path", names=[alias(name="join", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.join"] == "os.path.join"

def test_imports_with_import_from_node_and_asname():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="os.path", names=[alias(name="join", asname="path_join")], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.path_join"] == "os.path.join"

def test_imports_with_import_from_node_and_level():
    parser = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="sibling", names=[alias(name="func", asname=None)], level=1)
    parser.imports(root, node)
    assert parser.alias["test.module.submodule.func"] == "test.module.sibling.func"


# LLM-generated content at query #62
#--------------------------

```python
def test_class_api_predicate_false():
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)
    assert parser.doc[name] == ""


# LLM-generated content at query #63
#--------------------------

```python
def test_compile_empty_parser():
    parser = Parser.new(link=False, level=1, toc=False)
    assert parser.compile() == '\n'

def test_compile_with_basic_module():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', 'def func():\n    pass')
    result = parser.compile()
    assert '# Module `test_module`' in result
    assert '<a id="test-module"></a>' in result
    assert 'func()' in result
    assert '**Table of contents:**' in result

def test_compile_with_non_public_items():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
def _private_func():
    pass

def public_func():
    pass
''')
    result = parser.compile()
    assert 'public_func()' in result
    assert '_private_func' not in result

def test_compile_with_magic_method():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
class MyClass:
    def __init__(self):
        pass
''')
    result = parser.compile()
    assert 'class MyClass' in result
    assert '__init__' not in result

def test_compile_with_toc_disabled():
    parser = Parser.new(link=True, level=1, toc=False)
    parser.parse('test_module', 'def func():\n    pass')
    result = parser.compile()
    assert '**Table of contents:**' not in result
    assert 'func()' in result

def test_compile_with_link_disabled():
    parser = Parser.new(link=False, level=1, toc=True)
    parser.parse('test_module', 'def func():\n    pass')
    result = parser.compile()
    assert '<a id="test-module"></a>' not in result
    assert 'func()' in result

def test_compile_with_constants():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
MAX_SIZE = 100
def func():
    pass
''')
    result = parser.compile()
    assert 'MAX_SIZE' in result
    assert 'Constants' in result

def test_compile_with_class_members():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
class MyClass:
    def __init__(self):
        self.value: int = 0
''')
    result = parser.compile()
    assert 'class MyClass' in result
    assert 'Members' in result

def test_compile_with_inheritance():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
class Parent:
    pass

class Child(Parent):
    pass
''')
    result = parser.compile()
    assert 'class Parent' in result
    assert 'class Child' in result
    assert 'Bases' in result

def test_compile_with_decorators():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
def decorator(func):
    return func

@decorator
def func():
    pass
''')
    result = parser.compile()
    assert 'func()' in result
    assert 'Decorators' in result

def test_compile_with_nested_classes():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
class Outer:
    class Inner:
        pass
''')
    result = parser.compile()
    assert 'class Outer' in result
    assert 'class Inner' in result

def test_compile_with_enum():
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse('test_module', '''
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
''')
    result = parser.compile()
    assert 'class Color' in result
    assert 'Enums' in result
    assert 'RED' in result


# LLM-generated content at query #64
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments(posonlyargs=[arg('x', None)], args=[arg('y', None)], defaults=[])
    returns = None
    parser.func_api('root', 'root.func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == """### func()
*Full name:* `root.func`
<a id="root-func"></a>

| x | y | return |
|:---:|:---:|:---:|
| `Any` | `Any` | `Any` |
"""

def test_func_api_with_keyword_args():
    parser = Parser()
    node = arguments(args=[], kwonlyargs=[arg('x', None)], kw_defaults=[])
    returns = None
    parser.func_api('root', 'root.func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == """### func()
*Full name:* `root.func`
<a id="root-func"></a>

| * | x | return |
|:---:|:---:|:---:|
|  | `Any` | `Any` |
"""

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments(args=[], vararg=arg('args', None))
    returns = None
    parser.func_api('root', 'root.func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == """### func()
*Full name:* `root.func`
<a id="root-func"></a>

| *args | return |
|:---:|:---:|
| `Any` | `Any` |
"""

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments(args=[], kwarg=arg('kwargs', None))
    returns = None
    parser.func_api('root', 'root.func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == """### func()
*Full name:* `root.func`
<a id="root-func"></a>

| **kwargs | return |
|:---:|:---:|
| `Any` | `Any` |
"""

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(args=[arg('x', None), arg('y', None)], defaults=[Constant(value=1)])
    returns = None
    parser.func_api('root', 'root.func', node, returns, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == """### func()
*Full name:* `root.func`
<a id="root-func"></a>

| x | y | return |
|:---:|:---:|:---:|
| `Any` | `int` | `Any` |
|  | `1` |  |
"""

def test_func_api_with_self():
    parser = Parser()
    node = arguments(args=[arg('self', None), arg('x', None)], defaults=[])
    returns = None
    parser.func_api('root', 'root.Class.func', node, returns, has_self=True, cls_method=False)
    assert parser.doc['root.Class.func'] == """### func()
*Full name:* `root.Class.func`
<a id="root-class-func"></a>

| self | x | return |
|:---:|:---:|:---:|
| `Self` | `Any` | `Any` |
"""

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments(args=[arg('cls', None), arg('x', None)], defaults=[])
    returns = None
    parser.func_api('root', 'root.Class.func', node, returns, has_self=True, cls_method=True)
    assert parser.doc['root.Class.func'] == """### func()
*Full name:* `root.Class.func`
<a id="root-class-func"></a>

| cls | x | return |
|:---:|:---:|:---:|
| `type[Self]` | `Any` | `Any` |
"""


# LLM-generated content at query #65
#--------------------------

```python
def test_globals_with_ann_assign():
    node = AnnAssign(
        target=Name(id="test_var"),
        annotation=Name(id="int"),
        value=Constant(value=42)
    )
    parser = Parser()
    parser.globals("test.module", node)
    assert parser.alias["test.module.test_var"] == "42"
    assert parser.const["test.module.test_var"] == "int"

def test_globals_with_assign():
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42)
    )
    parser = Parser()
    parser.globals("test.module", node)
    assert parser.alias["test.module.test_var"] == "42"
    assert parser.const["test.module.test_var"] == "int"

def test_globals_with_type_comment():
    node = Assign(
        targets=[Name(id="test_var")],
        value=Constant(value=42),
        type_comment="int"
    )
    parser = Parser()
    parser.globals("test.module", node)
    assert parser.alias["test.module.test_var"] == "42"
    assert parser.const["test.module.test_var"] == "int"

def test_globals_with_all():
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="test_var")])
    )
    parser = Parser()
    parser.globals("test.module", node)
    assert parser.imp["test.module"] == {"test.module.test_var"}

def test_globals_with_non_constant():
    node = Assign(
        targets=[Name(id="test_var")],
        value=Name(id="some_var")
    )
    parser = Parser()
    parser.globals("test.module", node)
    assert parser.alias["test.module.test_var"] == "some_var"
    assert parser.const.get("test.module.test_var") == "Any"


# LLM-generated content at query #66
#--------------------------

```python
def test_func_ann_with_self_and_cls_method():
    parser = Parser()
    args = [arg('self', None), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_self_no_cls_method():
    parser = Parser()
    args = [arg('self', None), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_without_self():
    parser = Parser()
    args = [arg('x', Name(id='int', ctx=Load())), arg('y', Name(id='str', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', 'str']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('x', Name(id='int', ctx=Load())), arg('*', None), arg('y', Name(id='str', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']

def test_func_ann_with_no_annotation():
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['Any', 'Any']

def test_func_ann_with_self_type_annotation():
    parser = Parser()
    args = [arg('self', Name(id='MyClass', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method_and_self_type():
    parser = Parser()
    args = [arg('cls', Name(id='type[MyClass]', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']


# LLM-generated content at query #67
#--------------------------

```python
def test_compile_skips_magic_names():
    p = Parser()
    p.doc = {'__init__': '## Module `{}`\n\n'}
    p.root = {'__init__': '__init__'}
    p.level = {'__init__': 0}
    p.docstring = {}
    p.imp = {}
    p.toc = False
    result = p.compile()
    assert result == ''


# LLM-generated content at query #68
#--------------------------

```python
def test_visit_Name_self_ty_match():
    resolver = Resolver(root="", alias={}, self_ty="Test")
    node = Name(id="Test", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #69
#--------------------------

```python
def test__attr_returns_none_for_nonexistent_nested_attribute():
    class MockObject:
        pass
    obj = MockObject()
    assert _attr(obj, "nonexistent_attr") is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=5),
        simple=1
    )
    parser.globals("module", node)
    assert parser.alias["module.x"] == "5"
    assert parser.const["module.x"] == "int"
    assert parser.root["module.x"] == "module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value=3.14)
    )
    parser.globals("module", node)
    assert parser.alias["module.y"] == "3.14"
    assert parser.const["module.y"] == "float"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("module", node)
    assert parser.imp["module"] == {"module.func1", "module.func2"}

def test_globals_with_non_public_constant():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="_private", ctx=Store()),
        annotation=Name(id="str", ctx=Load()),
        value=Constant(value="secret"),
        simple=1
    )
    parser.globals("module", node)
    assert parser.alias["module._private"] == "'secret'"
    assert "module._private" not in parser.const

def test_globals_with_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=Constant(value=True),
        type_comment="bool"
    )
    parser.globals("module", node)
    assert parser.alias["module.z"] == "True"
    assert parser.const["module.z"] == "bool"


# LLM-generated content at query #2
#--------------------------

```python
def test_parser_default_constructor():
    parser = Parser()
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is False
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_custom_constructor():
    parser = Parser(link=False, b_level=2, toc=True)
    assert parser.link is False
    assert parser.b_level == 2
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link is False
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_toc_true():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #3
#--------------------------

```python
def test_const_type_constant():
    assert const_type(Constant(5)) == "int"
    assert const_type(Constant(3.14)) == "float"
    assert const_type(Constant("hello")) == "str"

def test_const_type_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant(1), Constant("a")])) == "tuple[Any, Any]"

def test_const_type_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int]"
    assert const_type(List([Constant(1), Constant("a")])) == "list[Any]"

def test_const_type_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int]"
    assert const_type(Set([Constant(1), Constant("a")])) == "set[Any]"

def test_const_type_dict():
    assert const_type(Dict([Constant(1)], [Constant("a")])) == "dict[int, str]"
    assert const_type(Dict([Constant(1), Constant(2)], [Constant("a"), Constant("b")])) == "dict[int, str]"
    assert const_type(Dict([Constant(1), Constant("a")], [Constant(1), Constant(2)])) == "dict[Any, Any]"

def test_const_type_call():
    assert const_type(Call(Name("bool", None, None), [], [])) == "bool"
    assert const_type(Call(Name("int", None, None), [], [])) == "int"
    assert const_type(Call(Name("float", None, None), [], [])) == "float"
    assert const_type(Call(Name("complex", None, None), [], [])) == "complex"
    assert const_type(Call(Name("str", None, None), [], [])) == "str"

def test_const_type_any():
    assert const_type(Call(Name("unknown", None, None), [], [])) == "Any"
    assert const_type(Constant(None)) == "Any"
    assert const_type(Tuple([])) == "tuple[]"
    assert const_type(List([])) == "list[]"
    assert const_type(Set([])) == "set[]"
    assert const_type(Dict([], [])) == "dict[]"


# LLM-generated content at query #4
#--------------------------

```python
def test_imports_with_regular_import():
    p = Parser()
    node = Import(names=[alias(name='sys', asname=None)])
    p.imports('pkg', node)
    assert p.alias['pkg.sys'] == 'sys'

def test_imports_with_asname():
    p = Parser()
    node = Import(names=[alias(name='sys', asname='system')])
    p.imports('pkg', node)
    assert p.alias['pkg.system'] == 'sys'

def test_imports_with_from_import():
    p = Parser()
    node = ImportFrom(module='sys', names=[alias(name='exit', asname=None)], level=0)
    p.imports('pkg', node)
    assert p.alias['pkg.exit'] == 'sys.exit'

def test_imports_with_from_import_and_asname():
    p = Parser()
    node = ImportFrom(module='sys', names=[alias(name='exit', asname='quit')], level=0)
    p.imports('pkg', node)
    assert p.alias['pkg.quit'] == 'sys.exit'

def test_imports_with_relative_import():
    p = Parser()
    node = ImportFrom(module='subpkg', names=[alias(name='func', asname=None)], level=1)
    p.imports('pkg.subpkg', node)
    assert p.alias['pkg.subpkg.func'] == 'pkg.subpkg.func'


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = """
\"\"\"Module docstring.\"\"\"
x = 1
def foo():
    \"\"\"Function docstring.\"\"\"
    pass
"""
    parser.parse("test", script)
    assert "test" in parser.doc
    assert "test.x" in parser.alias
    assert "test.foo" in parser.doc
    assert "test.foo" in parser.docstring

def test_parse_with_imports():
    parser = Parser()
    script = """
import os
from sys import path
x = 1
"""
    parser.parse("test", script)
    assert "test" in parser.doc
    assert "os" in parser.alias["test.os"]
    assert "sys.path" in parser.alias["test.path"]
    assert "test.x" in parser.alias

def test_parse_with_class_and_method():
    parser = Parser()
    script = """
class MyClass:
    \"\"\"Class docstring.\"\"\"
    def method(self):
        \"\"\"Method docstring.\"\"\"
        pass
"""
    parser.parse("test", script)
    assert "test.MyClass" in parser.doc
    assert "test.MyClass.method" in parser.doc
    assert "test.MyClass" in parser.docstring
    assert "test.MyClass.method" in parser.docstring

def test_parse_with_decorators():
    parser = Parser()
    script = """
@decorator
def foo():
    \"\"\"Function docstring.\"\"\"
    pass
"""
    parser.parse("test", script)
    assert "test.foo" in parser.doc
    assert "@decorator" in parser.doc["test.foo"]
    assert "test.foo" in parser.docstring

def test_parse_with_constants():
    parser = Parser()
    script = """
CONSTANT = 42
"""
    parser.parse("test", script)
    assert "test.CONSTANT" in parser.alias
    assert "test.CONSTANT" in parser.const
    assert "test.CONSTANT" in parser.root

def test_parse_with_all_filter():
    parser = Parser()
    script = """
__all__ = ["public_func"]
def public_func():
    pass
def _private_func():
    pass
"""
    parser.parse("test", script)
    assert "test.public_func" in parser.imp["test"]
    assert "test._private_func" not in parser.imp["test"]

def test_parse_with_nested_class():
    parser = Parser()
    script = """
class Outer:
    class Inner:
        pass
"""
    parser.parse("test", script)
    assert "test.Outer" in parser.doc
    assert "test.Outer.Inner" in parser.doc

def test_parse_with_try_except():
    parser = Parser()
    script = """
try:
    x = 1
except Exception:
    y = 2
else:
    z = 3
finally:
    w = 4
"""
    parser.parse("test", script)
    assert "test.x" in parser.alias
    assert "test.y" in parser.alias
    assert "test.z" in parser.alias
    assert "test.w" in parser.alias

def test_parse_with_if_else():
    parser = Parser()
    script = """
if True:
    x = 1
else:
    y = 2
"""
    parser.parse("test", script)
    assert "test.x" in parser.alias
    assert "test.y" in parser.alias

def test_parse_with_annotations():
    parser = Parser()
    script = """
x: int = 1
def foo(a: str) -> None:
    pass
"""
    parser.parse("test", script)
    assert "test.x" in parser.alias
    assert "test.foo" in parser.doc
    assert "int" in parser.const["test.x"]


# LLM-generated content at query #6
#--------------------------

```python
def test_parser_default_constructor():
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    parser = Parser.new(link=False, level=3, toc=True)
    assert parser.link == True  # toc overrides link
    assert parser.b_level == 3
    assert parser.toc == True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root.public_name': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root.public_name': 'root', 'root.other': 'root'}
    assert parser.is_public('root.public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root._private': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root._private': 'root', 'root.other': 'root'}
    assert parser.is_public('root._private') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': {'__init__'}}
    parser.doc = {'root.__init__': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root.__init__': 'root', 'root.other': 'root'}
    assert parser.is_public('root.__init__') == True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root.public_name.nested': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root.public_name.nested': 'root', 'root.other': 'root'}
    assert parser.is_public('root.public_name.nested') == True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root.public_name._nested': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root.public_name._nested': 'root', 'root.other': 'root'}
    assert parser.is_public('root.public_name._nested') == False

def test_is_public_with_empty_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root.public_name': '', 'root.other': ''}
    parser.const = {}
    parser.root = {'root.public_name': 'root', 'root.other': 'root'}
    assert parser.is_public('root.public_name') == True

def test_is_public_with_const_in_all():
    parser = Parser()
    parser.imp = {'root': {'CONST_NAME'}}
    parser.doc = {}
    parser.const = {'root.CONST_NAME': 'int'}
    parser.root = {'root.CONST_NAME': 'root'}
    assert parser.is_public('root.CONST_NAME') == True


# LLM-generated content at query #8
#--------------------------

```python
def test__e_type_empty_input():
    assert _e_type() == ""

def test__e_type_none_element():
    assert _e_type([None]) == ""

def test__e_type_mixed_constant_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_single_constant_type():
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_multiple_sequences():
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"

def test__e_type_non_constant_element():
    assert _e_type([Constant(1), 2]) == ""


# LLM-generated content at query #9
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "self_ty")
    node = Name("self_ty", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias.value"}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "value"

def test_visit_Name_without_alias():
    resolver = Resolver("root", {}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver("root", {"root.name": "typing.TypeVar('T')"}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #10
#--------------------------

```python
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

def test_parser_custom_constructor():
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is True  # toc overrides link
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #11
#--------------------------

```python
def test__defaults_empty_sequence():
    assert list(_defaults([])) == []

def test__defaults_none_values():
    assert list(_defaults([None, None])) == [" ", " "]

def test__defaults_with_values():
    assert list(_defaults([parse_expr("x"), parse_expr("y")])) == ["`x`", "`y`"]

def test__defaults_with_and_without_values():
    assert list(_defaults([None, parse_expr("x"), None])) == [" ", "`x`", " "]

def test__defaults_with_ampersand():
    assert list(_defaults([parse_expr("a & b")])) == ["<code>a &#38; b</code>"]

def test__defaults_with_pipe():
    assert list(_defaults([parse_expr("a | b")])) == ["<code>a &#124; b</code>"]

def test__defaults_with_empty_string():
    assert list(_defaults([parse_expr("")])) == [" "]


# LLM-generated content at query #12
#--------------------------

```python
def test_func_api_with_posonlyargs_and_defaults():
    parser = Parser()
    node = arguments(
        posonlyargs=[arg('a', None), arg('b', None)],
        args=[arg('c', None)],
        defaults=[Constant(value=1)],
        kwonlyargs=[],
        kw_defaults=[],
        vararg=None,
        kwarg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert '| a | b | / | c | return |' in parser.doc['root.func']
    assert '| --- | --- | --- | --- | --- |' in parser.doc['root.func']
    assert '| `a` | `b` |  | `c` | `Any` |' in parser.doc['root.func']
    assert '|  |  |  | `1` |  |' in parser.doc['root.func']

def test_func_api_with_vararg_and_kwarg():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[arg('b', None)],
        kw_defaults=[],
        vararg=arg('args', None),
        kwarg=arg('kwargs', None)
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert '| a | *args | b | **kwargs | return |' in parser.doc['root.func']
    assert '| --- | --- | --- | --- | --- |' in parser.doc['root.func']
    assert '| `a` | `Any` | `b` | `Any` | `Any` |' in parser.doc['root.func']

def test_func_api_with_self_and_cls_method():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('self', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        vararg=None,
        kwarg=None
    )
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    assert '| self | a | return |' in parser.doc['root.func']
    assert '| --- | --- | --- |' in parser.doc['root.func']
    assert '| `type[Self]` | `a` | `Any` |' in parser.doc['root.func']

def test_func_api_with_annotations():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('a', Name(id='int', ctx=Load())), arg('b', Name(id='str', ctx=Load()))],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        vararg=None,
        kwarg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert '| a | b | return |' in parser.doc['root.func']
    assert '| --- | --- | --- |' in parser.doc['root.func']
    assert '| `int` | `str` | `Any` |' in parser.doc['root.func']

def test_func_api_with_returns():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        vararg=None,
        kwarg=None
    )
    returns = Name(id='int', ctx=Load())
    parser.func_api('root', 'func', node, returns, has_self=False, cls_method=False)
    assert '| a | return |' in parser.doc['root.func']
    assert '| --- | --- |' in parser.doc['root.func']
    assert '| `a` | `int` |' in parser.doc['root.func']


# LLM-generated content at query #13
#--------------------------

```python
def test_func_api_with_no_args_and_no_return():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n|  |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_single_arg_and_return():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('x', None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    parser.func_api('root', 'func', node, Name(id='int', ctx=Load()), has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `x` | `int` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('x', None), arg('y', None)], kwonlyargs=[], kw_defaults=[], defaults=[Constant(value=1)])
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `x` | `y` | `return` |\n|  | `1` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_self_and_cls_method():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('self', None)], kwonlyargs=[], kw_defaults=[], defaults=[])
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    expected = "| arg | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_kwonlyargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[arg('x', None)], kw_defaults=[Constant(value=2)], defaults=[])
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `x` |  |\n| `2` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_vararg_and_kwarg():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[], vararg=arg('args', None), kwarg=arg('kwargs', None))
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| arg | return |\n|:---:|:---:|\n| `*args` | `**kwargs` |  |\n\n"
    assert parser.doc['root.func'] == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_imports_with_asname():
    p = Parser()
    node = Import(names=[alias(name='module', asname='alias')])
    p.imports('root', node)
    assert p.alias.get('root.alias') == 'module'


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    parser = Parser()
    parser.const = {"test_name": "some_value"}
    assert not (parser.const.get("test_name", "ANY") == "ANY")


# LLM-generated content at query #16
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="x"),
        annotation=Name(id="int"),
        value=Constant(value=1)
    )
    parser.globals("module", node)
    assert parser.alias["module.x"] == "1"
    assert parser.const["module.x"] == "int"
    assert parser.root["module.x"] == "module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y")],
        value=Constant(value="hello")
    )
    parser.globals("module", node)
    assert parser.alias["module.y"] == "'hello'"
    assert parser.const["module.y"] == "str"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="foo"), Constant(value="bar")])
    )
    parser.globals("module", node)
    assert parser.imp["module"] == {"module.foo", "module.bar"}

def test_globals_with_non_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="z")],
        value=Name(id="some_var")
    )
    parser.globals("module", node)
    assert parser.alias["module.z"] == "some_var"
    assert "module.z" not in parser.const


# LLM-generated content at query #17
#--------------------------

```python
def test_func_ann_with_self_and_annotation():
    parser = Parser()
    args = [arg('self', Name(id='MyClass', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method():
    parser = Parser()
    args = [arg('cls', Name(id='MyClass', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_without_annotation():
    parser = Parser()
    args = [arg('x', None), arg('y', Name(id='str', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['ANY', 'str']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('x', Name(id='int', ctx=Load())), arg('*', None), arg('y', Name(id='str', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']


# LLM-generated content at query #18
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=1))
    parser.globals('root', node)
    assert 'root.x' not in parser.alias


# LLM-generated content at query #19
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []

    parser.class_api(root, name, bases, body)

    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)

    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="VALUE1", ctx=Store()), annotation=Constant(value=1), value=Constant(value=1)),
        AnnAssign(target=Name(id="VALUE2", ctx=Store()), annotation=Constant(value=2), value=Constant(value=2))
    ]

    parser.class_api(root, name, bases, body)

    assert "Enums" in parser.doc[name]
    assert "VALUE1" in parser.doc[name]
    assert "VALUE2" in parser.doc[name]

def test_class_api_with_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="member1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        AnnAssign(target=Name(id="member2", ctx=Store()), annotation=Name(id="str", ctx=Load()), value=None)
    ]

    parser.class_api(root, name, bases, body)

    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]
    assert "member1" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "member2" in parser.doc[name]
    assert "str" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="member1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Delete(targets=[Name(id="member1", ctx=Del())])
    ]

    parser.class_api(root, name, bases, body)

    assert "member1" not in parser.doc[name]


# LLM-generated content at query #20
#--------------------------

```python
def test_const_type_with_constant_int():
    node = Constant(value=42)
    assert const_type(node) == "int"

def test_const_type_with_constant_str():
    node = Constant(value="hello")
    assert const_type(node) == "str"

def test_const_type_with_tuple_of_ints():
    node = Tuple(elts=[Constant(value=1), Constant(value=2)])
    assert const_type(node) == "tuple[int, int]"

def test_const_type_with_list_of_strs():
    node = List(elts=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "list[str, str]"

def test_const_type_with_set_of_floats():
    node = Set(elts=[Constant(value=1.1), Constant(value=2.2)])
    assert const_type(node) == "set[float, float]"

def test_const_type_with_dict_int_to_str():
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")])
    assert const_type(node) == "dict[int, int, str, str]"

def test_const_type_with_mixed_types_in_tuple():
    node = Tuple(elts=[Constant(value=1), Constant(value="a")])
    assert const_type(node) == "tuple[Any, Any]"

def test_const_type_with_empty_tuple():
    node = Tuple(elts=[])
    assert const_type(node) == "tuple[]"

def test_const_type_with_none_in_list():
    node = List(elts=[Constant(value=1), None])
    assert const_type(node) == ""

def test_const_type_with_call_to_bool():
    node = Call(func=Name(id="bool"), args=[])
    assert const_type(node) == "bool"

def test_const_type_with_call_to_int():
    node = Call(func=Name(id="int"), args=[])
    assert const_type(node) == "int"

def test_const_type_with_unknown_call():
    node = Call(func=Name(id="unknown"), args=[])
    assert const_type(node) == "Any"

def test_const_type_with_non_constant_in_list():
    node = List(elts=[Constant(value=1), Name(id="x")])
    assert const_type(node) == ""

def test_const_type_with_attribute_call():
    node = Call(func=Attribute(value=Name(id="obj"), attr="method"), args=[])
    assert const_type(node) == "Any"


# LLM-generated content at query #21
#--------------------------

```python
def test_class_api():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test_module', '''
class TestClass(BaseClass):
    """Test class docstring."""
    attr1: int
    attr2: str = "default"
    _private_attr: bool = True
    __magic_attr__: float = 1.0

    def __init__(self):
        pass
''')
    assert 'TestClass' in parser.doc
    assert 'Bases' in parser.doc['test_module.TestClass']
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'Type' in parser.doc['test_module.TestClass']
    assert 'attr1' in parser.doc['test_module.TestClass']
    assert 'attr2' in parser.doc['test_module.TestClass']
    assert '_private_attr' not in parser.doc['test_module.TestClass']
    assert '__magic_attr__' not in parser.doc['test_module.TestClass']


# LLM-generated content at query #22
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id="x"), Name(id="y")], value=Constant(value=1))
    parser.globals("root", node)
    assert "root.x" not in parser.alias


# LLM-generated content at query #23
#--------------------------

```python
def test_attr_simple_attribute():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, 'x') == 10

def test_attr_nested_attribute():
    class InnerClass:
        y = 20
    class OuterClass:
        inner = InnerClass()
    obj = OuterClass()
    assert _attr(obj, 'inner.y') == 20

def test_attr_nonexistent_attribute():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, 'z') is None

def test_attr_nonexistent_nested_attribute():
    class InnerClass:
        y = 20
    class OuterClass:
        inner = InnerClass()
    obj = OuterClass()
    assert _attr(obj, 'inner.z') is None

def test_attr_missing_intermediate_attribute():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, 'inner.y') is None

def test_attr_empty_string():
    class TestClass:
        x = 10
    obj = TestClass()
    assert _attr(obj, '') == obj


# LLM-generated content at query #24
#--------------------------

```python
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(
        target=Name(id="var"),
        annotation=Name(id="int"),
        value=Constant(value=5)
    )
    p.globals("module", node)
    assert p.alias["module.var"] == "5"
    assert p.const["module.var"] == "int"
    assert p.root["module.var"] == "module"

def test_globals_with_assign():
    p = Parser()
    node = Assign(
        targets=[Name(id="var")],
        value=Constant(value=5)
    )
    p.globals("module", node)
    assert p.alias["module.var"] == "5"
    assert p.const["module.var"] == "int"

def test_globals_with_type_comment():
    p = Parser()
    node = Assign(
        targets=[Name(id="var")],
        value=Constant(value=5),
        type_comment="int"
    )
    p.globals("module", node)
    assert p.alias["module.var"] == "5"
    assert p.const["module.var"] == "int"

def test_globals_with__all__():
    p = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func")])
    )
    p.globals("module", node)
    assert p.imp["module"] == {"module.func"}

def test_globals_ignores_complex_assign():
    p = Parser()
    node = Assign(
        targets=[Name(id="var1"), Name(id="var2")],
        value=Constant(value=5)
    )
    p.globals("module", node)
    assert "module.var1" not in p.alias
    assert "module.var2" not in p.alias


# LLM-generated content at query #25
#--------------------------

```python
def test_compile_empty():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == '\n'

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc['test'] = '#' * 2 + " Module `{}`"
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.imp['test'] = set()
    assert p.compile() == '**Table of contents:**\n\n+ [test](#test)\n\n## Module `test`\n'

def test_compile_with_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '#' * 2 + " Module `{}`"
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.imp['test'] = set()
    p.docstring['test'] = "This is a test module."
    assert p.compile() == '## Module `test`\n\nThis is a test module.\n'

def test_compile_with_const():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test'] = '#' * 2 + " Module `{}`"
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.imp['test'] = set()
    p.const['test.CONST'] = 'int'
    assert p.compile() == '## Module `test`\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n'

def test_compile_with_magic():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc['test.__init__'] = '#' * 3 + " `{}`()\n\n"
    p.level['test.__init__'] = 0
    p.root['test.__init__'] = 'test'
    p.imp['test'] = set()
    assert p.compile() == '\n'

def test_compile_with_link():
    p = Parser.new(link=True, level=1, toc=False)
    p.doc['test'] = '#' * 2 + " Module `{}`\n<a id=\"{}\"></a>\n\n"
    p.level['test'] = 0
    p.root['test'] = 'test'
    p.imp['test'] = set()
    assert p.compile() == '## Module `test`\n<a id="test"></a>\n\n'


# LLM-generated content at query #26
#--------------------------

```python
def test_imports_with_Import_node():
    p = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname=None)])
    p.imports(root, node)
    assert p.alias["test.module.os"] == "os"

def test_imports_with_Import_node_and_asname():
    p = Parser()
    root = "test.module"
    node = Import(names=[alias(name="os", asname="operating_system")])
    p.imports(root, node)
    assert p.alias["test.module.operating_system"] == "os"

def test_imports_with_ImportFrom_node_with_level():
    p = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    p.imports(root, node)
    assert p.alias["test.module.submodule.path"] == "test.module.os.path"

def test_imports_with_ImportFrom_node_without_level():
    p = Parser()
    root = "test.module"
    node = ImportFrom(module="os", names=[alias(name="path", asname="ospath")], level=0)
    p.imports(root, node)
    assert p.alias["test.module.ospath"] == "os.path"


# LLM-generated content at query #27
#--------------------------

```python
def test_is_magic_predicate():
    p = Parser()
    p.doc = {'__init__': '', '__add__': ''}
    p.root = {'__init__': '__init__', '__add__': '__add__'}
    p.imp = {}
    p.docstring = {}
    p.level = {'__init__': 0, '__add__': 0}
    p.toc = False
    result = p.compile()
    assert result == ''


# LLM-generated content at query #28
#--------------------------

```python
def test_api_function():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), returns=None, body=[])
    parser.api(root, node)
    assert "test_func" in parser.doc
    assert "test_module.test_func" in parser.doc
    assert parser.doc["test_module.test_func"].startswith("### test_func()")

def test_api_async_function():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="async_func", args=arguments(), returns=None, body=[])
    parser.api(root, node)
    assert "async_func" in parser.doc
    assert "test_module.async_func" in parser.doc
    assert parser.doc["test_module.async_func"].startswith("### async async_func()")

def test_api_class():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], body=[])
    parser.api(root, node)
    assert "TestClass" in parser.doc
    assert "test_module.TestClass" in parser.doc
    assert parser.doc["test_module.TestClass"].startswith("### class TestClass")

def test_api_with_prefix():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="method", args=arguments(), returns=None, body=[])
    parser.api(root, node, prefix="TestClass")
    assert "TestClass.method" in parser.doc
    assert "test_module.TestClass.method" in parser.doc
    assert parser.doc["test_module.TestClass.method"].startswith("#### method()")

def test_api_with_decorators():
    parser = Parser()
    root = "test_module"
    decorator = Name(id="staticmethod", ctx=Load())
    node = FunctionDef(name="decorated_func", args=arguments(), returns=None, body=[], decorator_list=[decorator])
    parser.api(root, node)
    assert "decorated_func" in parser.doc
    assert "test_module.decorated_func" in parser.doc
    assert "@staticmethod" in parser.doc["test_module.decorated_func"]

def test_api_with_docstring():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="doc_func", args=arguments(), returns=None, body=[], decorator_list=[])
    node.body = [Expr(value=Constant(value="This is a docstring"))]
    parser.api(root, node)
    assert "doc_func" in parser.doc
    assert "test_module.doc_func" in parser.doc
    assert "This is a docstring" in parser.docstring["test_module.doc_func"]

def test_api_class_with_bases():
    parser = Parser()
    root = "test_module"
    base = Name(id="BaseClass", ctx=Load())
    node = ClassDef(name="DerivedClass", bases=[base], body=[])
    parser.api(root, node)
    assert "DerivedClass" in parser.doc
    assert "test_module.DerivedClass" in parser.doc
    assert "Bases" in parser.doc["test_module.DerivedClass"]
    assert "BaseClass" in parser.doc["test_module.DerivedClass"]

def test_api_class_with_members():
    parser = Parser()
    root = "test_module"
    ann_assign = AnnAssign(target=Name(id="member"), annotation=Name(id="int", ctx=Load()), value=None)
    node = ClassDef(name="MemberClass", bases=[], body=[ann_assign])
    parser.api(root, node)
    assert "MemberClass" in parser.doc
    assert "test_module.MemberClass" in parser.doc
    assert "Members" in parser.doc["test_module.MemberClass"]
    assert "member" in parser.doc["test_module.MemberClass"]
    assert "int" in parser.doc["test_module.MemberClass"]

def test_api_class_with_enum():
    parser = Parser()
    root = "test_module"
    base = Name(id="enum.Enum", ctx=Load())
    assign = Assign(targets=[Name(id="VALUE1", ctx=Store())], value=Constant(value=1))
    node = ClassDef(name="EnumClass", bases=[base], body=[assign])
    parser.api(root, node)
    assert "EnumClass" in parser.doc
    assert "test_module.EnumClass" in parser.doc
    assert "Enums" in parser.doc["test_module.EnumClass"]
    assert "VALUE1" in parser.doc["test_module.EnumClass"]


# LLM-generated content at query #29
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Attribute_keeps_non_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(Name("other", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "List"

def test_visit_Attribute_returns_node_if_value_not_Name():
    resolver = Resolver(root="", alias={})
    node = Attribute(Constant("not_a_name"), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert isinstance(result.value, Constant)


# LLM-generated content at query #30
#--------------------------

```python
def test_attr_single_level():
    class TestObj:
        def __init__(self):
            self.value = 42
    obj = TestObj()
    assert _attr(obj, 'value') == 42

def test_attr_multi_level():
    class TestObj:
        def __init__(self):
            self.nested = NestedObj()
    class NestedObj:
        def __init__(self):
            self.value = 42
    obj = TestObj()
    assert _attr(obj, 'nested.value') == 42

def test_attr_nonexistent():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, 'nonexistent') is None

def test_attr_partial_nonexistent():
    class TestObj:
        def __init__(self):
            self.nested = NestedObj()
    class NestedObj:
        pass
    obj = TestObj()
    assert _attr(obj, 'nested.nonexistent') is None

def test_attr_empty_string():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, '') == obj


# LLM-generated content at query #31
#--------------------------

```python
def test_class_api_with_enum_bases():
    p = Parser()
    p.parse('test_module', '''
from enum import Enum
class TestClass(Enum):
    A = 1
    B = 2
''')
    assert 'Enums' in p.doc['test_module.TestClass']
    assert 'Members' not in p.doc['test_module.TestClass']


# LLM-generated content at query #32
#--------------------------

```python
def test_visit_constant_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) is node

def test_visit_constant_invalid_syntax():
    resolver = Resolver("root", {})
    node = Constant("invalid syntax !@#")
    assert resolver.visit_Constant(node) is node

def test_visit_constant_valid_name():
    resolver = Resolver("root", {"root.name": "alias"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"

def test_visit_constant_valid_name_with_self():
    resolver = Resolver("root", {}, "Self")
    node = Constant("Self")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "Self"


# LLM-generated content at query #33
#--------------------------

```python
def test_api_function():
    parser = Parser()
    root = "test"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test.test_func" in parser.doc
    assert parser.doc["test.test_func"] == "### test_func()\n\n*Full name:* `test.test_func`\n<a id=\"test-test_func\"></a>\n\n"

def test_api_async_function():
    parser = Parser()
    root = "test"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test.test_async_func" in parser.doc
    assert parser.doc["test.test_async_func"] == "### async test_async_func()\n\n*Full name:* `test.test_async_func`\n<a id=\"test-test_async_func\"></a>\n\n"

def test_api_class():
    parser = Parser()
    root = "test"
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "test.TestClass" in parser.doc
    assert parser.doc["test.TestClass"] == "### class TestClass\n\n*Full name:* `test.TestClass`\n<a id=\"test-TestClass\"></a>\n\n"

def test_api_with_prefix():
    parser = Parser()
    root = "test"
    node = FunctionDef(name="nested_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node, prefix="OuterClass")
    assert "test.OuterClass.nested_func" in parser.doc
    assert parser.doc["test.OuterClass.nested_func"] == "#### nested_func()\n\n*Full name:* `test.OuterClass.nested_func`\n<a id=\"test-OuterClass-nested_func\"></a>\n\n"

def test_api_with_decorators():
    parser = Parser()
    root = "test"
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="decorated_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[decorator])
    parser.api(root, node)
    assert "test.decorated_func" in parser.doc
    assert parser.doc["decorated_func"].startswith("### decorated_func()\n\n*Full name:* `test.decorated_func`\n<a id=\"test-decorated_func\"></a>\n\n| Decorators |\n|:-----------:|\n| @decorator |")

def test_api_with_docstring():
    parser = Parser()
    root = "test"
    node = FunctionDef(name="doc_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[], body=[Expr(value=Constant(value="This is a docstring"))])
    parser.api(root, node)
    assert "test.doc_func" in parser.doc
    assert parser.doc["doc_func"] == "### doc_func()\n\n*Full name:* `test.doc_func`\n<a id=\"test-doc_func\"></a>\n\n"
    assert "test.doc_func" in parser.docstring
    assert parser.docstring["test.doc_func"] == "```python\nThis is a docstring\n```"


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_line_38_evaluates_to_false():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=List(elts=[Constant(value=123)]))
    p.globals('root', node)
    assert '__all__' not in p.imp


# LLM-generated content at query #35
#--------------------------

```python
def test_visit_Constant_non_string_value():
    resolver = Resolver(root="", alias={})
    node = Constant(value=123)
    assert resolver.visit_Constant(node) == node


# LLM-generated content at query #36
#--------------------------

```python
def test_func_api_with_vararg():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    node = arguments(
        posonlyargs=[arg(arg="a", annotation=None)],
        args=[arg(arg="b", annotation=None)],
        defaults=[Constant(value=1)],
        vararg=arg(arg="args", annotation=None),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None
    )
    returns = None
    has_self = False
    cls_method = False
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert True


# LLM-generated content at query #37
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Attribute_non_typing_prefix():
    resolver = Resolver("root", {})
    node = Attribute(Name("other", Load()), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"
    assert result.value.id == "other"

def test_visit_Attribute_non_name_value():
    resolver = Resolver("root", {})
    node = Attribute(Constant("not a name"), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"


# LLM-generated content at query #38
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments(posonlyargs=[arg('x', None), arg('y', None)], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| x | y | return |\n|:---:|:---:|:---:|\n|  |  |  |\n\n"

def test_func_api_with_keyword_args():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[arg('a', None), arg('b', None)], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| a | b | return |\n|:---:|:---:|:---:|\n|  |  |  |\n\n"

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=arg('args', None))
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| *args | return |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| **kwargs | return |\n|:---:|:---:|\n|  |  |\n\n"

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('x', None)], defaults=[Constant(value=1)], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| x | return |\n|:---:|:---:|\n| `1` |  |\n|  |  |\n\n"

def test_func_api_with_self():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('self', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=True, cls_method=False)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| self | return |\n|:---:|:---:|\n| `Self` |  |\n\n"

def test_func_api_with_classmethod():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('cls', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'test_func', node, None, has_self=True, cls_method=True)
    assert parser.doc['root.test_func'] == "# test_func()\n\n*Full name:* `root.test_func`\n\n| cls | return |\n|:---:|:---:|\n| `type[Self]` |  |\n\n"


# LLM-generated content at query #39
#--------------------------

```python
def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('*', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == [""]


# LLM-generated content at query #40
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser.new(link=False, level=1, toc=False)
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    parser.class_api('root.module', 'root.module.ClassName', bases, body)
    assert parser.doc['root.module.ClassName'] == '#' * 3 + " class ClassName\n\n*Full name:* `{}`\n\n"
    assert parser.doc['root.module.ClassName'] == '#' * 3 + " class ClassName\n\n*Full name:* `{}`\n\n" + table("Bases", items=['BaseClass'])

def test_class_api_with_enums():
    parser = Parser.new(link=False, level=1, toc=False)
    bases = [Name(id='enum.Enum', ctx=Load())]
    body = [AnnAssign(target=Name(id='RED', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)]
    parser.class_api('root.module', 'root.module.Color', bases, body)
    assert parser.doc['root.module.Color'] == '#' * 3 + " class Color\n\n*Full name:* `{}`\n\n"
    assert parser.doc['root.module.Color'] == '#' * 3 + " class Color\n\n*Full name:* `{}`\n\n" + table("Enums", items=['RED'])

def test_class_api_with_members():
    parser = Parser.new(link=False, level=1, toc=False)
    bases = []
    body = [AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1)]
    parser.class_api('root.module', 'root.module.ClassName', bases, body)
    assert parser.doc['root.module.ClassName'] == '#' * 3 + " class ClassName\n\n*Full name:* `{}`\n\n"
    assert parser.doc['root.module.ClassName'] == '#' * 3 + " class ClassName\n\n*Full name:* `{}`\n\n" + table('Members', 'Type', items=[('attr1', 'int')])

def test_class_api_with_deleted_members():
    parser = Parser.new(link=False, level=1, toc=False)
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None, simple=1),
        Delete(targets=[Name(id='attr1', ctx=Del())])
    ]
    parser.class_api('root.module', 'root.module.ClassName', bases, body)
    assert parser.doc['root.module.ClassName'] == '#' * 3 + " class ClassName\n\n*Full name:* `{}`\n\n"
    assert 'attr1' not in parser.doc['root.module.ClassName']


# LLM-generated content at query #41
#--------------------------

```python
def test_const_type_with_constant():
    assert const_type(Constant(5)) == "int"
    assert const_type(Constant(3.14)) == "float"
    assert const_type(Constant("hello")) == "str"
    assert const_type(Constant(True)) == "bool"

def test_const_type_with_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant(1), Constant("a")])) == "tuple[Any, Any]"
    assert const_type(Tuple([])) == "tuple[]"

def test_const_type_with_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int]"
    assert const_type(List([Constant(1), Constant("a")])) == "list[Any]"
    assert const_type(List([])) == "list[]"

def test_const_type_with_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int]"
    assert const_type(Set([Constant(1), Constant("a")])) == "set[Any]"
    assert const_type(Set([])) == "set[]"

def test_const_type_with_dict():
    assert const_type(Dict([Constant(1)], [Constant("a")])) == "dict[int, str]"
    assert const_type(Dict([Constant(1), Constant("a")], [Constant(2), Constant("b")])) == "dict[Any, Any]"
    assert const_type(Dict([], [])) == "dict[]"

def test_const_type_with_call():
    assert const_type(Call(Name("bool"), [])) == "bool"
    assert const_type(Call(Name("int"), [])) == "int"
    assert const_type(Call(Name("float"), [])) == "float"
    assert const_type(Call(Name("complex"), [])) == "complex"
    assert const_type(Call(Name("str"), [])) == "str"
    assert const_type(Call(Attribute(Name("x"), "y"), [])) == "ANY"


# LLM-generated content at query #42
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'


# LLM-generated content at query #43
#--------------------------

```python
def test_class_api():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test_module', '''
class TestClass(BaseClass):
    """Test class docstring."""
    x: int
    y = 1
    def method(self):
        pass
''')
    parser.class_api('test_module.TestClass', 'test_module.TestClass', [Name(id='BaseClass')], [])
    assert 'Bases' in parser.doc['test_module.TestClass']
    assert 'Members' in parser.doc['test_module.TestClass']
    assert 'Type' in parser.doc['test_module.TestClass']


# LLM-generated content at query #44
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(
        targets=[Name(id='__all__')],
        value=Tuple(elts=[Constant(value='foo')], ctx=Load())
    )
    parser.globals('root', node)
    assert '__all__' not in parser.imp['root']


# LLM-generated content at query #45
#--------------------------

```python
def test_func_api_with_posonlyargs():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[arg('a', None), arg('b', None)],
        args=[arg('c', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| a | b | / | c | return |\n|:---:|:---:|:---:|:---:|:---:|\n| `str` | `str` |  | `str` | `str` |\n\n'

def test_func_api_with_vararg():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=arg('args', None)
    )
    parser.func_api('root', 'root.func', args, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| a | *args | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n\n'

def test_func_api_with_kwonlyargs():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[arg('a', None), arg('b', None)],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `str` | `str` | `str` |\n\n'

def test_func_api_with_kwarg():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg('kwargs', None),
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| **kwargs | return |\n|:---:|:---:|\n| `str` | `str` |\n\n'

def test_func_api_with_defaults():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[arg('a', None), arg('b', None)],
        defaults=[Constant(value=1), Constant(value=2)],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| a | b | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n| `1` | `2` |  |\n\n'

def test_func_api_with_has_self():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[arg('self', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=True, cls_method=False)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| Self | a | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n\n'

def test_func_api_with_cls_method():
    parser = Parser.new(link=False, level=1, toc=False)
    args = arguments(
        posonlyargs=[],
        args=[arg('cls', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', args, None, has_self=True, cls_method=True)
    assert parser.doc['root.func'] == '#' * 3 + ' func()\n\n*Full name:* `{}`\n\n| type[Self] | a | return |\n|:---:|:---:|:---:|\n| `str` | `str` | `str` |\n\n'


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name(id="test_attr")],
            value=Constant(value=42),
            type_comment=None
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "test_attr" not in parser.doc[name]


# LLM-generated content at query #47
#--------------------------

```python
def test_func_api_with_kwonlyargs_and_no_vararg():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    node = arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[arg("kwarg1", None), arg("kwarg2", None)],
        kw_defaults=[],
        defaults=[],
        vararg=None,
        kwarg=None
    )
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    assert parser.doc[name].endswith("\n\n| * | kwarg1 | kwarg2 | return |\n|---|---|---|---|\n|  | ANY | ANY | ANY |\n")


# LLM-generated content at query #48
#--------------------------

```python
def test_load_docstring():
    parser = Parser()
    parser.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'func()'}
    parser.docstring = {}

    module = ModuleType('pkg.module')
    module.func = lambda: None
    module.func.__doc__ = "This is a function."

    parser.load_docstring('pkg.module', module)

    assert parser.docstring['pkg.module.func'] == "```python\nThis is a function.\n```"


# LLM-generated content at query #49
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    args = arguments(posonlyargs=[arg('x', None)], args=[arg('y', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    has_self = False
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| x | y | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_keyword_args():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[arg('x', None)], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    has_self = False
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| * | x | return |\n|:---:|:---:|:---:|\n|  | `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_varargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=arg('args', None))
    returns = None
    has_self = False
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_kwargs():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), vararg=None)
    returns = None
    has_self = False
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_defaults():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('x', None)], defaults=[Constant(value=1)], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    has_self = False
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| x | return |\n|:---:|:---:|\n| `Any` | `Any` |\n| `1` |  |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_self():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('self', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    has_self = True
    cls_method = False
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| Self | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected

def test_func_api_with_cls_method():
    parser = Parser()
    args = arguments(posonlyargs=[], args=[arg('cls', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    has_self = False
    cls_method = True
    parser.func_api('root', 'root.func', args, returns, has_self=has_self, cls_method=cls_method)
    expected = "| type[Self] | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == "| ### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n" + expected


# LLM-generated content at query #50
#--------------------------

```python
def test_is_public_family_with_public_name():
    assert is_public_family('public') == True

def test_is_public_family_with_magic_name():
    assert is_public_family('__init__') == True

def test_is_public_family_with_private_name():
    assert is_public_family('_private') == False

def test_is_public_family_with_local_name():
    assert is_public_family('module._local') == False

def test_is_public_family_with_mixed_names():
    assert is_public_family('public.__magic__._private') == False

def test_is_public_family_with_nested_public_names():
    assert is_public_family('public1.public2.public3') == True

def test_is_public_family_with_empty_string():
    assert is_public_family('') == True

def test_is_public_family_with_single_underscore():
    assert is_public_family('_') == False

def test_is_public_family_with_double_underscore_prefix():
    assert is_public_family('__public') == True

def test_is_public_family_with_double_underscore_suffix():
    assert is_public_family('public__') == False


