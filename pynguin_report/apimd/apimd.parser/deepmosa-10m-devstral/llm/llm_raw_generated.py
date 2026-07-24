####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports_with_import_node():
    parser = Parser()
    root = "test.module"
    node = Import(names=[alias(name="sys", asname=None), alias(name="os", asname="operating_system")])
    parser.imports(root, node)
    assert parser.alias["test.module.sys"] == "sys"
    assert parser.alias["test.module.operating_system"] == "os"

def test_imports_with_import_from_node_no_level():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="sys", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.path"] == "sys.path"

def test_imports_with_import_from_node_with_level():
    parser = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="sibling", names=[alias(name="helper", asname=None)], level=1)
    parser.imports(root, node)
    assert parser.alias["test.module.submodule.helper"] == "test.module.sibling.helper"


# LLM-generated content at query #2
#--------------------------

```python
def test__defaults_with_none_values():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_empty_strings():
    args = ["", "", ""]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test__defaults_with_ampersand():
    args = ["a&b", "c&d"]
    result = list(_defaults(args))
    assert result == ["<code>a&b</code>", "<code>c&d</code>"]

def test__defaults_with_pipe():
    args = ["a|b", "c|d"]
    result = list(_defaults(args))
    assert result == ["`a&#124;b`", "`c&#124;d`"]

def test__defaults_with_mixed_values():
    args = ["a", None, "b&c", "d|e"]
    result = list(_defaults(args))
    assert result == ["`a`", " ", "<code>b&c</code>", "`d&#124;e`"]


# LLM-generated content at query #3
#--------------------------

```python
def test__attr_basic():
    class A:
        x = 1
    assert _attr(A, 'x') == 1

def test__attr_nested():
    class A:
        class B:
            y = 2
        b = B()
    assert _attr(A, 'b.y') == 2

def test__attr_nonexistent():
    class A:
        x = 1
    assert _attr(A, 'z') is None

def test__attr_nonexistent_nested():
    class A:
        class B:
            y = 2
        b = B()
    assert _attr(A, 'b.z') is None

def test__attr_chain_break():
    class A:
        class B:
            y = 2
        b = B()
    assert _attr(A, 'b.z.w') is None


# LLM-generated content at query #4
#--------------------------

```python
def test_globals_with_ann_assign():
    p = Parser()
    node = AnnAssign(target=Name(id='x'), annotation=Name(id='int'), value=Constant(value=1))
    p.globals('root', node)
    assert p.alias['root.x'] == '1'
    assert p.const['root.x'] == 'int'
    assert p.root['root.x'] == 'root'

def test_globals_with_assign():
    p = Parser()
    node = Assign(targets=[Name(id='y')], value=Constant(value=2.5))
    p.globals('root', node)
    assert p.alias['root.y'] == '2.5'
    assert p.const['root.y'] == 'float'

def test_globals_with_all():
    p = Parser()
    node = Assign(targets=[Name(id='__all__')], value=List(elts=[Constant(value='foo'), Constant(value='bar')]))
    p.globals('root', node)
    assert p.imp['root'] == {'root.foo', 'root.bar'}

def test_globals_with_non_uppercase():
    p = Parser()
    node = Assign(targets=[Name(id='z')], value=Constant(value='hello'))
    p.globals('root', node)
    assert p.alias['root.z'] == "'hello'"
    assert 'root.z' not in p.const


# LLM-generated content at query #5
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp['module'] = {'public_name'}
    parser.root['module.public_name'] = 'module'
    parser.doc['module.public_name'] = ''
    assert parser.is_public('module.public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp['module'] = {'public_name'}
    parser.root['module._private_name'] = 'module'
    assert parser.is_public('module._private_name') == False

def test_is_public_with_listed_in_all():
    parser = Parser()
    parser.imp['module'] = {'listed_name'}
    parser.root['module.listed_name'] = 'module'
    assert parser.is_public('module.listed_name') == True

def test_is_public_with_parent_listed_in_all():
    parser = Parser()
    parser.imp['module'] = {'parent_name'}
    parser.root['module.parent_name.child'] = 'module'
    assert parser.is_public('module.parent_name.child') == True

def test_is_public_with_empty_all():
    parser = Parser()
    parser.imp['module'] = set()
    parser.root['module.public_name'] = 'module'
    parser.doc['module.public_name'] = ''
    assert parser.is_public('module.public_name') == True

def test_is_public_with_non_public_family():
    parser = Parser()
    parser.imp['module'] = set()
    parser.root['module._private_name'] = 'module'
    assert parser.is_public('module._private_name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp['module'] = set()
    parser.root['module.__magic__'] = 'module'
    parser.doc['module.__magic__'] = ''
    assert parser.is_public('module.__magic__') == True


# LLM-generated content at query #6
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

def test_parser_new_method():
    p = Parser.new(link=False, level=3, toc=True)
    assert p.link is True  # toc=True forces link=True
    assert p.b_level == 3
    assert p.toc is True
    assert p.level == {}
    assert p.doc == {}
    assert p.docstring == {}
    assert p.imp == {}
    assert p.root == {}
    assert p.alias == {}
    assert p.const == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_class_api_basic():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root["test"] = "test"
    parser.level["test"] = 0
    parser.doc["test"] = "# test\n\n"
    parser.alias["test"] = "test"
    parser.const["test"] = "test"

    class_def = ClassDef(
        name="TestClass",
        bases=[Name(id="Base", ctx=Load())],
        keywords=[],
        body=[],
        decorator_list=[]
    )

    parser.class_api("test", "test.TestClass", class_def.bases, class_def.body)

    expected_doc = "# test\n\n*Full name:* `test.TestClass`\n\n| Bases |\n|:---:|\n| `Base` |\n\n"
    assert parser.doc["test.TestClass"] == expected_doc

def test_class_api_with_members():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root["test"] = "test"
    parser.level["test"] = 0
    parser.doc["test"] = "# test\n\n"
    parser.alias["test"] = "test"
    parser.const["test"] = "test"

    assign = Assign(
        targets=[Name(id="member", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )

    class_def = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[assign],
        decorator_list=[]
    )

    parser.class_api("test", "test.TestClass", class_def.bases, class_def.body)

    expected_doc = "# test\n\n*Full name:* `test.TestClass`\n\n| Members | Type |\n|:---:|:---:|\n| `member` | `int` |\n\n"
    assert parser.doc["test.TestClass"] == expected_doc

def test_class_api_with_enum():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root["test"] = "test"
    parser.level["test"] = 0
    parser.doc["test"] = "# test\n\n"
    parser.alias["test"] = "test"
    parser.const["test"] = "test"

    assign1 = Assign(
        targets=[Name(id="ENUM1", ctx=Store())],
        value=Constant(value=1),
        type_comment=None
    )

    assign2 = Assign(
        targets=[Name(id="ENUM2", ctx=Store())],
        value=Constant(value=2),
        type_comment=None
    )

    class_def = ClassDef(
        name="TestEnum",
        bases=[Name(id="enum.Enum", ctx=Load())],
        keywords=[],
        body=[assign1, assign2],
        decorator_list=[]
    )

    parser.class_api("test", "test.TestEnum", class_def.bases, class_def.body)

    expected_doc = "# test\n\n*Full name:* `test.TestEnum`\n\n| Bases |\n|:---:|\n| `enum.Enum` |\n\n| Enums |\n|:---:|\n| `ENUM1` |\n| `ENUM2` |\n\n"
    assert parser.doc["test.TestEnum"] == expected_doc

def test_class_api_with_deleted_member():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root["test"] = "test"
    parser.level["test"] = 0
    parser.doc["test"] = "# test\n\n"
    parser.alias["test"] = "test"
    parser.const["test"] = "test"

    assign = Assign(
        targets=[Name(id="member", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )

    delete = Delete(targets=[Name(id="member", ctx=Del())])

    class_def = ClassDef(
        name="TestClass",
        bases=[],
        keywords=[],
        body=[assign, delete],
        decorator_list=[]
    )

    parser.class_api("test", "test.TestClass", class_def.bases, class_def.body)

    expected_doc = "# test\n\n*Full name:* `test.TestClass`\n\n"
    assert parser.doc["test.TestClass"] == expected_doc


# LLM-generated content at query #8
#--------------------------

```python
def test__attr_returns_none_for_invalid_attribute_path():
    class MockObject:
        pass

    obj = MockObject()
    assert _attr(obj, "invalid.attr.path") is None


# LLM-generated content at query #9
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias.value"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "value"

def test_visit_Name_without_alias():
    resolver = Resolver("root", {}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver("root", {"root.name": "typing.TypeVar('T')"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #10
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

def test_parser_constructor_with_custom_values():
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

def test_parser_post_init_toc_sets_link():
    p = Parser(toc=True)
    assert p.link is True
    assert p.toc is True


# LLM-generated content at query #11
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = ast.Attribute(value=ast.Name(id='typing', ctx=ast.Load()), attr='List', ctx=ast.Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, ast.Name)
    assert result.id == 'List'

def test_visit_Attribute_keeps_non_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = ast.Attribute(value=ast.Name(id='other', ctx=ast.Load()), attr='List', ctx=ast.Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, ast.Attribute)
    assert result.value.id == 'other'
    assert result.attr == 'List'


# LLM-generated content at query #12
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "| BaseClass |" in parser.doc[name]

def test_class_api_without_bases():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]

def test_class_api_with_enum_members():
    parser = Parser()
    root = "module"
    name = "module.EnumClass"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="MEMBER1", ctx=Store()), annotation=Constant(value=1), value=None),
        AnnAssign(target=Name(id="MEMBER2", ctx=Store()), annotation=Constant(value=2), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "| MEMBER1 |" in parser.doc[name]
    assert "| MEMBER2 |" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Assign(targets=[Name(id="another_attr", ctx=Store())], value=Constant(value=42))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "| public_attr |" in parser.doc[name]
    assert "| int |" in parser.doc[name]
    assert "| another_attr |" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "module"
    name = "module.Class"
    bases = []
    body = [
        AnnAssign(target=Name(id="_private_attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None)
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #13
#--------------------------

```python
def test_is_public_family_with_public_name():
    assert is_public_family('module.submodule.function') == True

def test_is_public_family_with_private_name():
    assert is_public_family('module._private') == False

def test_is_public_family_with_magic_name():
    assert is_public_family('module.__init__') == True

def test_is_public_family_with_local_name():
    assert is_public_family('_local') == False

def test_is_public_family_with_mixed_names():
    assert is_public_family('module._private.public') == False

def test_is_public_family_with_empty_string():
    assert is_public_family('') == True

def test_is_public_family_with_single_public_name():
    assert is_public_family('public') == True

def test_is_public_family_with_single_private_name():
    assert is_public_family('_private') == False

def test_is_public_family_with_single_magic_name():
    assert is_public_family('__magic__') == True


# LLM-generated content at query #14
#--------------------------

```python
def test_parse_basic_module():
    parser = Parser()
    script = """
    '''Module docstring.'''
    def foo():
        '''Function docstring.'''
        pass
    """
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.foo" in parser.docstring

def test_parse_with_imports():
    parser = Parser()
    script = """
    import os
    from sys import path
    def bar():
        pass
    """
    parser.parse("test_module", script)
    assert "os" in parser.alias
    assert "path" in parser.alias
    assert "test_module.bar" in parser.doc

def test_parse_with_assignments():
    parser = Parser()
    script = """
    CONST = 42
    x: int = 10
    """
    parser.parse("test_module", script)
    assert "test_module.CONST" in parser.const
    assert "test_module.x" in parser.alias

def test_parse_with_class():
    parser = Parser()
    script = """
    class MyClass:
        '''Class docstring.'''
        def method(self):
            pass
    """
    parser.parse("test_module", script)
    assert "test_module.MyClass" in parser.doc
    assert "test_module.MyClass.method" in parser.doc
    assert "test_module.MyClass" in parser.docstring

def test_parse_with_decorators():
    parser = Parser()
    script = """
    @decorator
    def decorated_func():
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.decorated_func" in parser.doc
    assert "@decorator" in parser.doc["test_module.decorated_func"]

def test_parse_with_try_except():
    parser = Parser()
    script = """
    try:
        def inner_func():
            pass
    except:
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.inner_func" in parser.doc

def test_parse_with_nested_classes():
    parser = Parser()
    script = """
    class Outer:
        class Inner:
            pass
    """
    parser.parse("test_module", script)
    assert "test_module.Outer" in parser.doc
    assert "test_module.Outer.Inner" in parser.doc

def test_parse_with_ann_assign():
    parser = Parser()
    script = """
    x: int = 5
    y: str
    """
    parser.parse("test_module", script)
    assert "test_module.x" in parser.alias
    assert "test_module.y" not in parser.alias

def test_parse_with_all_filter():
    parser = Parser()
    script = """
    __all__ = ['public_func']
    def public_func():
        pass
    def _private_func():
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.public_func" in parser.doc
    assert "test_module._private_func" not in parser.doc

def test_parse_with_link_option():
    parser = Parser(link=True)
    script = "def func(): pass"
    parser.parse("test_module", script)
    assert "<a id=\"test_module.func\"></a>" in parser.doc["test_module.func"]

def test_parse_with_toc_option():
    parser = Parser(toc=True)
    script = "def func(): pass"
    parser.parse("test_module", script)
    assert parser.link is True


# LLM-generated content at query #15
#--------------------------

```python
def test_const_type_constant():
    assert const_type(Constant(1)) == "int"
    assert const_type(Constant(1.0)) == "float"
    assert const_type(Constant("hello")) == "str"
    assert const_type(Constant(True)) == "bool"

def test_const_type_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant(1), Constant(2.0)])) == "tuple[Any, Any]"
    assert const_type(Tuple([Constant(1), Constant("hello")])) == "tuple[Any, Any]"
    assert const_type(Tuple([])) == "tuple[]"

def test_const_type_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int, int]"
    assert const_type(List([Constant(1), Constant(2.0)])) == "list[Any, Any]"
    assert const_type(List([Constant(1), Constant("hello")])) == "list[Any, Any]"
    assert const_type(List([])) == "list[]"

def test_const_type_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int, int]"
    assert const_type(Set([Constant(1), Constant(2.0)])) == "set[Any, Any]"
    assert const_type(Set([Constant(1), Constant("hello")])) == "set[Any, Any]"
    assert const_type(Set([])) == "set[]"

def test_const_type_dict():
    assert const_type(Dict([Constant(1), Constant("a")], [Constant(2), Constant("b")])) == "dict[int, int, str, str]"
    assert const_type(Dict([Constant(1), Constant(2.0)], [Constant("a"), Constant("b")])) == "dict[Any, Any, str, str]"
    assert const_type(Dict([], [])) == "dict[]"

def test_const_type_call():
    assert const_type(Call(Name("bool"), [])) == "bool"
    assert const_type(Call(Name("int"), [])) == "int"
    assert const_type(Call(Name("float"), [])) == "float"
    assert const_type(Call(Name("complex"), [])) == "complex"
    assert const_type(Call(Name("str"), [])) == "str"
    assert const_type(Call(Name("list"), [])) == "list"
    assert const_type(Call(Name("dict"), [])) == "dict"
    assert const_type(Call(Name("unknown"), [])) == "Any"

def test_const_type_other():
    assert const_type(Name("x")) == "Any"
    assert const_type(Attribute(Name("x"), "y")) == "Any"


# LLM-generated content at query #16
#--------------------------

```python
def test_parse_assign_node():
    p = Parser()
    script = "x = 1"
    p.parse("test", script)
    assert "test" in p.doc


# LLM-generated content at query #17
#--------------------------

```python
def test_enums_not_empty():
    parser = Parser()
    parser.doc = {}
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.level = {}
    parser.imp = {}
    parser.b_level = 1
    parser.link = True
    parser.toc = False
    parser.docstring = {}

    class_node = ClassDef(
        name='TestClass',
        bases=[Name(id='Enum', ctx=Load())],
        body=[
            AnnAssign(
                target=Name(id='VALUE1', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=Constant(value=1)
            ),
            AnnAssign(
                target=Name(id='VALUE2', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=Constant(value=2)
            )
        ],
        decorator_list=[]
    )

    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)

    assert 'Enums' in parser.doc['test_module.TestClass']


# LLM-generated content at query #18
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id='x'), Name(id='y')], value=Constant(value=1))
    parser.globals('root', node)
    assert parser.alias == {}
    assert parser.root == {}
    assert parser.const == {}
    assert parser.imp == {'root': set()}


# LLM-generated content at query #19
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="test_var", ctx=Load()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

def test_globals_with_assign():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

def test_globals_with_assign_no_type_comment():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"
    assert parser.root["test_module.test_var"] == root

def test_globals_with__all__():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="public_func")])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func"}

def test_globals_with_non_constant__all__():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Name(id="public_func", ctx=Load())])
    )
    parser.globals(root, node)
    assert parser.imp[root] == set()

def test_globals_with_non_name_target():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Tuple(elts=[Name(id="test_var", ctx=Store())])],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert "test_module.test_var" not in parser.alias
    assert "test_module.test_var" not in parser.const
    assert "test_module.test_var" not in parser.root

def test_globals_with_non_uppercase_name():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert "test_module.test_var" not in parser.const
    assert "test_module.test_var" not in parser.root


# LLM-generated content at query #20
#--------------------------

```python
def test_class_api_enums():
    p = Parser()
    p.parse('pkg', 'class MyEnum(enum.Enum):\n    A = 1\n    B = 2')
    assert 'Enums' in p.doc['pkg.MyEnum']


# LLM-generated content at query #21
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id='typing', ctx=Load()), attr='List', ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == 'List'


# LLM-generated content at query #22
#--------------------------

```python
def test_api_function():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    parser.doc[root] = "# Module `test_module`\n\n"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()

    # Test function without decorator
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "test_func" in parser.doc
    assert parser.doc["test_func"] == "## test_func()\n\n*Full name:* `test_module.test_func`\n\n"

    # Test async function
    node = AsyncFunctionDef(name="async_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "async_func" in parser.doc
    assert parser.doc["async_func"] == "## async async_func()\n\n*Full name:* `test_module.async_func`\n\n"

    # Test class
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "TestClass" in parser.doc
    assert parser.doc["TestClass"] == "## class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"

def test_api_with_decorators():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    parser.doc[root] = "# Module `test_module`\n\n"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()

    # Test function with decorator
    node = FunctionDef(name="decorated_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[Name(id="decorator", ctx=Load())], returns=None)
    parser.api(root, node)
    assert "decorated_func" in parser.doc
    assert "Decorators" in parser.doc["decorated_func"]
    assert "decorator" in parser.doc["decorated_func"]

def test_api_with_docstring():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    parser.doc[root] = "# Module `test_module`\n\n"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()

    # Test function with docstring
    node = FunctionDef(name="doc_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Expr(value=Constant(value="This is a docstring"))], decorator_list=[], returns=None)
    parser.api(root, node)
    assert "doc_func" in parser.doc
    assert "This is a docstring" in parser.docstring["test_module.doc_func"]

def test_api_with_prefix():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    parser.doc[root] = "# Module `test_module`\n\n"
    parser.level[root] = 0
    parser.root[root] = root
    parser.imp[root] = set()

    # Test class method
    class_node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api(root, class_node)
    method_node = FunctionDef(name="method", args=arguments(posonlyargs=[], args=[arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[], returns=None)
    parser.api(root, method_node, prefix="TestClass")
    assert "TestClass.method" in parser.doc
    assert parser.doc["TestClass.method"] == "### method()\n\n*Full name:* `test_module.TestClass.method`\n\n"


# LLM-generated content at query #23
#--------------------------

```python
def test_func_ann_with_has_self_and_cls_method():
    parser = Parser()
    args = [arg('self', None), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_with_has_self_no_cls_method():
    parser = Parser()
    args = [arg('self', None), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('x', None), arg('*', None), arg('y', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['str', '', 'str']

def test_func_ann_with_no_annotation():
    parser = Parser()
    args = [arg('x', None)]
    result = list(parser.func_ann('root', args, has_self=False, cls_method=False))
    assert result == ['Any']

def test_func_ann_with_self_type_in_cls_method():
    parser = Parser()
    parser.alias['root.Self'] = 'root.Class'
    args = [arg('self', Name(id='Self', ctx=Load()))]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=True))
    assert result == ['type[Self]']


# LLM-generated content at query #24
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver("root", {}, "Self")
    node = Name("Self", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver("root", {"root.name": "alias"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "alias"

def test_visit_Name_without_alias():
    resolver = Resolver("root", {}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

def test_visit_Name_with_TypeVar():
    resolver = Resolver("root", {"root.name": "typing.TypeVar('T')"}, "Self")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #25
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

def test_parser_post_init_with_toc():
    parser = Parser(toc=True)
    assert parser.link is True
    assert parser.toc is True

def test_parser_post_init_without_toc():
    parser = Parser(toc=False)
    assert parser.link is True
    assert parser.toc is False


# LLM-generated content at query #26
#--------------------------

```python
def test_attr_simple_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.simple_attr = 42
    assert _attr(obj, 'simple_attr') == 42

def test_attr_nested_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.nested = TestClass()
    obj.nested.attr = 'value'
    assert _attr(obj, 'nested.attr') == 'value'

def test_attr_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, 'nonexistent') is None

def test_attr_nonexistent_nested_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.nested = TestClass()
    assert _attr(obj, 'nested.nonexistent') is None

def test_attr_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, '') is obj

def test_attr_chained_nonexistent():
    class TestClass:
        pass
    obj = TestClass()
    obj.level1 = TestClass()
    obj.level1.level2 = TestClass()
    assert _attr(obj, 'level1.level2.level3') is None


# LLM-generated content at query #27
#--------------------------

```python
def test_globals_with_ann_assign_and_constant():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="MY_CONST"),
        annotation=Name(id="int"),
        value=Constant(value=42)
    )
    parser.globals("test.module", node)
    assert parser.alias["test.module.MY_CONST"] == "42"
    assert parser.const["test.module.MY_CONST"] == "int"
    assert parser.root["test.module.MY_CONST"] == "test.module"

def test_globals_with_assign_and_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="MY_CONST")],
        value=Constant(value=42)
    )
    parser.globals("test.module", node)
    assert parser.alias["test.module.MY_CONST"] == "42"
    assert parser.const["test.module.MY_CONST"] == "int"

def test_globals_with_assign_and_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="MY_CONST")],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals("test.module", node)
    assert parser.alias["test.module.MY_CONST"] == "42"
    assert parser.const["test.module.MY_CONST"] == "int"

def test_globals_with_assign_and_list():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("test.module", node)
    assert parser.imp["test.module"] == {"test.module.func1", "test.module.func2"}

def test_globals_with_assign_and_tuple():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=Tuple(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("test.module", node)
    assert parser.imp["test.module"] == {"test.module.func1", "test.module.func2"}

def test_globals_with_assign_and_non_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="my_var")],
        value=Name(id="some_value")
    )
    parser.globals("test.module", node)
    assert parser.alias["test.module.my_var"] == "some_value"
    assert "test.module.my_var" not in parser.const

def test_globals_with_ann_assign_and_non_constant():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="my_var"),
        annotation=Name(id="int"),
        value=Name(id="some_value")
    )
    parser.globals("test.module", node)
    assert parser.alias["test.module.my_var"] == "some_value"
    assert "test.module.my_var" not in parser.const

def test_globals_with_assign_and_multiple_targets():
    parser = Parser()
    node = Assign(
        targets=[Name(id="my_var1"), Name(id="my_var2")],
        value=Constant(value=42)
    )
    parser.globals("test.module", node)
    assert "test.module.my_var1" not in parser.alias
    assert "test.module.my_var2" not in parser.alias


# LLM-generated content at query #28
#--------------------------

```python
def test_const_type_constant():
    assert const_type(Constant(5)) == "int"
    assert const_type(Constant(3.14)) == "float"
    assert const_type(Constant("hello")) == "str"

def test_const_type_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant(1), Constant(2.0)])) == "tuple[Any, Any]"
    assert const_type(Tuple([Constant(1), Constant("a")])) == "tuple[Any, Any]"

def test_const_type_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int, int]"
    assert const_type(List([Constant(1), Constant(2.0)])) == "list[Any, Any]"
    assert const_type(List([Constant(1), Constant("a")])) == "list[Any, Any]"

def test_const_type_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int, int]"
    assert const_type(Set([Constant(1), Constant(2.0)])) == "set[Any, Any]"
    assert const_type(Set([Constant(1), Constant("a")])) == "set[Any, Any]"

def test_const_type_dict():
    assert const_type(Dict([Constant(1)], [Constant("a")])) == "dict[int, str]"
    assert const_type(Dict([Constant(1)], [Constant(2.0)])) == "dict[int, float]"
    assert const_type(Dict([Constant(1), Constant(2)], [Constant("a"), Constant("b")])) == "dict[int, str]"

def test_const_type_call():
    assert const_type(Call(Name("bool"), [])) == "bool"
    assert const_type(Call(Name("int"), [])) == "int"
    assert const_type(Call(Attribute(Name("x"), "float"), [])) == "float"

def test_const_type_any():
    assert const_type(Name("x")) == "Any"
    assert const_type(Attribute(Name("x"), "y")) == "Any"


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_line_26_evaluates_to_false():
    parser = Parser()
    root = "test_module"
    name = "TestClass"
    bases = []
    body = [
        Assign(
            targets=[Name(id="public_attr")],
            value=Constant(value=42),
            type_comment=None
        )
    ]
    parser.class_api(root, name, bases, body)
    assert "public_attr" not in parser.doc[name]


# LLM-generated content at query #30
#--------------------------

```python
def test_func_api_with_posonlyargs():
    parser = Parser()
    node = arguments(posonlyargs=[arg('a', None), arg('b', None)], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=False, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | / | return |\n|:---:|:---:|:---:|:---:|\n| `Any` | `Any` |  | `Any` |\n\n'

def test_func_api_with_vararg():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=arg('args', None))
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=False, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

def test_func_api_with_kwonlyargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[arg('a', None), arg('b', None)], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=False, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n'

def test_func_api_with_kwarg():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=False, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n'

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('a', None), arg('b', None)], defaults=[Constant(1), Constant(2)], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=False, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n| `1` | `2` |  |\n\n'

def test_func_api_with_has_self():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('self', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=True, cls_method=False)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:---:|:---:|\n| `Self` | `Any` |\n\n'

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('cls', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    returns = None
    parser.func_api('root', 'name', node, returns, has_self=True, cls_method=True)
    assert parser.doc['name'] == '### name()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| cls | return |\n|:---:|:---:|\n| `type[Self]` | `Any` |\n\n'


# LLM-generated content at query #31
#--------------------------

```python
def test_api_with_function_def():
    parser = Parser()
    root_node = parse("def foo(): pass")
    node = root_node.body[0]
    parser.api("root", node)
    assert "root.foo" in parser.doc
    assert "## foo()" in parser.doc["root.foo"]

def test_api_with_async_function_def():
    parser = Parser()
    root_node = parse("async def bar(): pass")
    node = root_node.body[0]
    parser.api("root", node)
    assert "root.bar" in parser.doc
    assert "## async bar()" in parser.doc["root.bar"]

def test_api_with_class_def():
    parser = Parser()
    root_node = parse("class Baz: pass")
    node = root_node.body[0]
    parser.api("root", node)
    assert "root.Baz" in parser.doc
    assert "## class Baz" in parser.doc["root.Baz"]

def test_api_with_prefix():
    parser = Parser()
    root_node = parse("class Outer:\n    def inner(): pass")
    outer_node = root_node.body[0]
    inner_node = outer_node.body[0]
    parser.api("root", inner_node, prefix="Outer")
    assert "root.Outer.inner" in parser.doc
    assert "### inner()" in parser.doc["root.Outer.inner"]

def test_api_with_decorators():
    parser = Parser()
    root_node = parse("@decorator\ndef decorated(): pass")
    node = root_node.body[0]
    parser.api("root", node)
    assert "Decorators" in parser.doc["root.decorated"]
    assert "| @decorator |" in parser.doc["root.decorated"]

def test_api_with_docstring():
    parser = Parser()
    root_node = parse('def documented():\n    """This is a docstring."""\n    pass')
    node = root_node.body[0]
    parser.api("root", node)
    assert "root.documented" in parser.docstring
    assert "This is a docstring." in parser.docstring["root.documented"]

def test_api_with_class_body():
    parser = Parser()
    root_node = parse("class Container:\n    def method(): pass")
    node = root_node.body[0]
    parser.api("root", node)
    assert "root.Container" in parser.doc
    assert "root.Container.method" in parser.doc


# LLM-generated content at query #32
#--------------------------

```python
def test_api_function_without_prefix():
    parser = Parser.new(link=False, level=1, toc=False)
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api("root", node)
    assert "root.test_func" in parser.doc
    assert parser.doc["root.test_func"] == "### test_func()\n\n*Full name:* `root.test_func`\n\n"

def test_api_function_with_prefix():
    parser = Parser.new(link=False, level=1, toc=False)
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api("root", node, prefix="Class")
    assert "root.Class.test_func" in parser.doc
    assert parser.doc["root.Class.test_func"] == "#### test_func()\n\n*Full name:* `root.Class.test_func`\n\n"

def test_api_async_function():
    parser = Parser.new(link=False, level=1, toc=False)
    node = AsyncFunctionDef(name="test_async", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api("root", node)
    assert "root.test_async" in parser.doc
    assert parser.doc["root.test_async"] == "### async test_async()\n\n*Full name:* `root.test_async`\n\n"

def test_api_class():
    parser = Parser.new(link=False, level=1, toc=False)
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[], decorator_list=[])
    parser.api("root", node)
    assert "root.TestClass" in parser.doc
    assert parser.doc["root.TestClass"] == "### class TestClass\n\n*Full name:* `root.TestClass`\n\n"

def test_api_with_decorators():
    parser = Parser.new(link=False, level=1, toc=False)
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[decorator])
    parser.api("root", node)
    assert "root.test_func" in parser.doc
    assert "Decorators" in parser.doc["root.test_func"]
    assert "| decorator |" in parser.doc["root.test_func"]

def test_api_with_link_enabled():
    parser = Parser.new(link=True, level=1, toc=False)
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api("root", node)
    assert "root.test_func" in parser.doc
    assert "<a id=\"root-test_func\"></a>" in parser.doc["root.test_func"]

def test_api_class_with_bases():
    parser = Parser.new(link=False, level=1, toc=False)
    base = Name(id="BaseClass", ctx=Load())
    node = ClassDef(name="TestClass", bases=[base], keywords=[], body=[], decorator_list=[])
    parser.api("root", node)
    assert "root.TestClass" in parser.doc
    assert "Bases" in parser.doc["root.TestClass"]
    assert "| BaseClass |" in parser.doc["root.TestClass"]

def test_api_class_with_enums():
    parser = Parser.new(link=False, level=1, toc=False)
    enum_base = Name(id="enum.Enum", ctx=Load())
    node = ClassDef(name="TestEnum", bases=[enum_base], keywords=[], body=[
        AnnAssign(target=Name(id="VALUE1", ctx=Store()), annotation=Constant(value=1), value=Constant(value=1), simple=1),
        AnnAssign(target=Name(id="VALUE2", ctx=Store()), annotation=Constant(value=2), value=Constant(value=2), simple=1)
    ], decorator_list=[])
    parser.api("root", node)
    assert "root.TestEnum" in parser.doc
    assert "Enums" in parser.doc["root.TestEnum"]
    assert "| VALUE1 |" in parser.doc["root.TestEnum"]
    assert "| VALUE2 |" in parser.doc["root.TestEnum"]

def test_api_class_with_members():
    parser = Parser.new(link=False, level=1, toc=False)
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[
        AnnAssign(target=Name(id="member1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=1), simple=1),
        AnnAssign(target=Name(id="member2", ctx=Store()), annotation=Name(id="str", ctx=Load()), value=Constant(value="test"), simple=1)
    ], decorator_list=[])
    parser.api("root", node)
    assert "root.TestClass" in parser.doc
    assert "Members" in parser.doc["root.TestClass"]
    assert "| member1 | int |" in parser.doc["root.TestClass"]
    assert "| member2 | str |" in parser.doc["root.TestClass"]

def test_api_class_with_deleted_member():
    parser = Parser.new(link=False, level=1, toc=False)
    node = ClassDef(name="TestClass", bases=[], keywords=[], body=[
        AnnAssign(target=Name(id="member1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=1), simple=1),
        Delete(targets=[Name(id="member1", ctx=Del())])
    ], decorator_list=[])
    parser.api("root", node)
    assert "root.TestClass" in parser.doc
    assert "Members" not in parser.doc["root.TestClass"]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_globals_with_annassign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="VAR"),
        annotation=Name(id="int"),
        value=Constant(value=42)
    )
    parser.globals("module", node)
    assert parser.alias["module.VAR"] == "42"
    assert parser.const["module.VAR"] == "int"
    assert parser.root["module.VAR"] == "module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="VAR")],
        value=Constant(value=42)
    )
    parser.globals("module", node)
    assert parser.alias["module.VAR"] == "42"
    assert parser.const["module.VAR"] == "int"
    assert parser.root["module.VAR"] == "module"

def test_globals_with_type_comment():
    parser = Parser()
    node = Assign(
        targets=[Name(id="VAR")],
        value=Constant(value=42),
        type_comment="float"
    )
    parser.globals("module", node)
    assert parser.alias["module.VAR"] == "42"
    assert parser.const["module.VAR"] == "float"
    assert parser.root["module.VAR"] == "module"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("module", node)
    assert parser.imp["module"] == {"module.func1", "module.func2"}

def test_globals_ignores_non_constant():
    parser = Parser()
    node = Assign(
        targets=[Name(id="VAR")],
        value=Name(id="other_var")
    )
    parser.globals("module", node)
    assert "module.VAR" not in parser.alias
    assert "module.VAR" not in parser.const
    assert "module.VAR" not in parser.root


# LLM-generated content at query #2
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {}
    parser.doc = {'public_name': ''}
    parser.const = {}
    parser.root = {'public_name': 'public_name'}
    assert parser.is_public('public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {}
    parser.doc = {'_private_name': ''}
    parser.const = {}
    parser.root = {'_private_name': '_private_name'}
    assert parser.is_public('_private_name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {}
    parser.doc = {'__magic__': ''}
    parser.const = {}
    parser.root = {'__magic__': '__magic__'}
    assert parser.is_public('__magic__') == True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {}
    parser.doc = {'parent.public_child': ''}
    parser.const = {}
    parser.root = {'parent.public_child': 'parent'}
    assert parser.is_public('parent.public_child') == True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {}
    parser.doc = {'parent._private_child': ''}
    parser.const = {}
    parser.root = {'parent._private_child': 'parent'}
    assert parser.is_public('parent._private_child') == False

def test_is_public_with_all_listed_name():
    parser = Parser()
    parser.imp = {'parent': {'parent.child'}}
    parser.doc = {'parent.child': ''}
    parser.const = {}
    parser.root = {'parent.child': 'parent'}
    assert parser.is_public('parent.child') == True

def test_is_public_with_not_listed_in_all():
    parser = Parser()
    parser.imp = {'parent': {'parent.other'}}
    parser.doc = {'parent.child': ''}
    parser.const = {}
    parser.root = {'parent.child': 'parent'}
    assert parser.is_public('parent.child') == False

def test_is_public_with_const_in_all():
    parser = Parser()
    parser.imp = {'parent': {'parent.CONST'}}
    parser.doc = {}
    parser.const = {'parent.CONST': 'int'}
    parser.root = {'parent.CONST': 'parent'}
    assert parser.is_public('parent.CONST') == True


# LLM-generated content at query #3
#--------------------------

```python
def test__defaults_empty():
    assert list(_defaults([])) == [" "]

def test__defaults_none():
    assert list(_defaults([None])) == [" "]

def test__defaults_single_literal():
    assert list(_defaults(["x"])) == ["`x`"]

def test__defaults_multiple_literals():
    assert list(_defaults(["x", "y", "z"])) == ["`x`", "`y`", "`z`"]

def test__defaults_with_ampersand():
    assert list(_defaults(["x&y"])) == ["<code>x&#38;y</code>"]

def test__defaults_with_pipe():
    assert list(_defaults(["x|y"])) == ["`<code>x&#124;y</code>`"]

def test__defaults_mixed():
    assert list(_defaults(["x", None, "y&z", "a|b"])) == ["`x`", " ", "<code>y&#38;z</code>", "`<code>a&#124;b</code>`"]


# LLM-generated content at query #4
#--------------------------

```python
def test_api_function():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'def func(): pass')
    assert 'func' in parser.doc
    assert parser.doc['func'] == '### func()\n\n*Full name:* `test.func`\n\n'

def test_api_async_function():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'async def func(): pass')
    assert 'func' in parser.doc
    assert parser.doc['func'] == '### async func()\n\n*Full name:* `test.func`\n\n'

def test_api_class():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'class MyClass: pass')
    assert 'MyClass' in parser.doc
    assert parser.doc['MyClass'] == '### class MyClass\n\n*Full name:* `test.MyClass`\n\n'

def test_api_with_prefix():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'class MyClass:\n    def method(self): pass')
    assert 'MyClass.method' in parser.doc
    assert parser.doc['MyClass.method'] == '#### method()\n\n*Full name:* `test.MyClass.method`\n\n'

def test_api_with_decorator():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', '@decorator\ndef func(): pass')
    assert 'func' in parser.doc
    assert parser.doc['func'] == '### func()\n\n*Full name:* `test.func`\n\n|Decorators|\n|:---:|\n|`@decorator`|\n'

def test_api_with_docstring():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'def func():\n    """This is a docstring."""\n    pass')
    assert 'func' in parser.doc
    assert parser.doc['func'] == '### func()\n\n*Full name:* `test.func`\n\n'
    assert 'func' in parser.docstring
    assert parser.docstring['func'] == '```python\nThis is a docstring.\n```'

def test_api_nested_class():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.parse('test', 'class Outer:\n    class Inner: pass')
    assert 'Outer.Inner' in parser.doc
    assert parser.doc['Outer.Inner'] == '#### class Inner\n\n*Full name:* `test.Outer.Inner`\n\n'


# LLM-generated content at query #5
#--------------------------

```python
def test__e_type_empty_input():
    assert _e_type() == ""

def test__e_type_none_element():
    assert _e_type([None]) == ""

def test__e_type_non_constant_element():
    assert _e_type([1, 2]) == ""

def test__e_type_single_type():
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_mixed_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"

def test__e_type_multiple_sequences():
    assert _e_type([Constant(1)], [Constant(2)]) == "[int, int]"

def test__e_type_mixed_sequences():
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"

def test__e_type_all_any():
    assert _e_type([Constant(1), Constant("a")], [Constant(2), Constant(3.0)]) == "[Any, Any]"


# LLM-generated content at query #6
#--------------------------

```python
def test_parse_with_empty_script():
    parser = Parser()
    parser.parse("test_module", "")
    assert parser.doc["test_module"] == "### Module `{}`"
    assert parser.level["test_module"] == 0
    assert parser.imp["test_module"] == set()
    assert parser.root["test_module"] == "test_module"

def test_parse_with_docstring_only():
    parser = Parser()
    script = '''
    """
    This is a test module.
    """
    '''
    parser.parse("test_module", script)
    assert parser.doc["test_module"] == "### Module `{}`"
    assert parser.docstring["test_module"] == "```python\n    \"\"\"\n    This is a test module.\n    \"\"\"\n```"

def test_parse_with_import_statement():
    parser = Parser()
    script = '''
    import os
    from sys import path
    '''
    parser.parse("test_module", script)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.path"] == "sys.path"

def test_parse_with_assignment():
    parser = Parser()
    script = '''
    CONSTANT = 42
    '''
    parser.parse("test_module", script)
    assert parser.alias["test_module.CONSTANT"] == "42"
    assert parser.root["test_module.CONSTANT"] == "test_module"
    assert parser.const["test_module.CONSTANT"] == "int"

def test_parse_with_function_definition():
    parser = Parser()
    script = '''
    def test_function():
        pass
    '''
    parser.parse("test_module", script)
    assert parser.doc["test_module.test_function"] == "#### test_function()\n\n*Full name:* `{}`"
    assert parser.level["test_module.test_function"] == 0
    assert parser.root["test_module.test_function"] == "test_module"

def test_parse_with_class_definition():
    parser = Parser()
    script = '''
    class TestClass:
        pass
    '''
    parser.parse("test_module", script)
    assert parser.doc["test_module.TestClass"] == "#### class TestClass\n\n*Full name:* `{}`"
    assert parser.level["test_module.TestClass"] == 0
    assert parser.root["test_module.TestClass"] == "test_module"


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_true():
    elements = ([Constant(1), Constant(2)],)
    assert _e_type(*elements) == "[Int]"


# LLM-generated content at query #8
#--------------------------

```python
def test_walk_body_empty_sequence():
    assert list(walk_body([])) == []

def test_walk_body_single_statement():
    stmt = stmt()
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_multiple_statements():
    stmt1 = stmt()
    stmt2 = stmt()
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

def test_walk_body_if_statement():
    if_node = If(test=Name(id='x'), body=[stmt()], orelse=[stmt()])
    assert list(walk_body([if_node])) == [stmt(), stmt()]

def test_walk_body_nested_if_statements():
    inner_if = If(test=Name(id='y'), body=[stmt()], orelse=[])
    outer_if = If(test=Name(id='x'), body=[inner_if], orelse=[stmt()])
    assert list(walk_body([outer_if])) == [stmt(), stmt()]

def test_walk_body_try_statement():
    try_node = Try(
        body=[stmt()],
        handlers=[ExceptHandler(body=[stmt()])],
        orelse=[stmt()],
        finalbody=[stmt()]
    )
    assert list(walk_body([try_node])) == [stmt(), stmt(), stmt(), stmt()]

def test_walk_body_mixed_statements():
    if_node = If(test=Name(id='x'), body=[stmt()], orelse=[stmt()])
    try_node = Try(body=[stmt()], handlers=[], orelse=[], finalbody=[])
    assert list(walk_body([if_node, try_node])) == [stmt(), stmt(), stmt()]


# LLM-generated content at query #9
#--------------------------

```python
def test_empty_elements():
    assert _e_type() == ""


# LLM-generated content at query #10
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
    assert parser.link is True  # toc=True overrides link to True
    assert parser.b_level == 3
    assert parser.toc is True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = []

    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert parser.doc[name] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"

    bases = [Name(id="BaseClass", ctx=Load())]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "| BaseClass |" in parser.doc[name]

    body = [
        AnnAssign(
            target=Name(id="attr1", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        AnnAssign(
            target=Name(id="_private_attr", ctx=Store()),
            annotation=Name(id="str", ctx=Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name(id="attr2", ctx=Store())],
            value=Constant(value=42),
            type_comment="float"
        ),
        Delete(targets=[Name(id="attr1", ctx=Del())])
    ]
    parser.class_api(root, name, [], body)
    assert "Members" in parser.doc[name]
    assert "| attr2 | float |" in parser.doc[name]
    assert "| attr1 |" not in parser.doc[name]


# LLM-generated content at query #12
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver("module", {})
    node = Attribute(Name("typing", Load()), "List", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"

def test_visit_Attribute_non_typing_prefix():
    resolver = Resolver("module", {})
    node = Attribute(Name("other", Load()), "attr", Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.attr == "attr"


# LLM-generated content at query #13
#--------------------------

```python
def test_attr_simple_attribute():
    class MockObj:
        pass
    obj = MockObj()
    obj.value = 42
    assert _attr(obj, 'value') == 42

def test_attr_nested_attribute():
    class MockObj:
        pass
    obj = MockObj()
    obj.nested = MockObj()
    obj.nested.value = 42
    assert _attr(obj, 'nested.value') == 42

def test_attr_missing_attribute():
    class MockObj:
        pass
    obj = MockObj()
    assert _attr(obj, 'missing') is None

def test_attr_missing_nested_attribute():
    class MockObj:
        pass
    obj = MockObj()
    obj.nested = MockObj()
    assert _attr(obj, 'nested.missing') is None

def test_attr_empty_string():
    class MockObj:
        pass
    obj = MockObj()
    assert _attr(obj, '') is None

def test_attr_none_object():
    assert _attr(None, 'value') is None


# LLM-generated content at query #14
#--------------------------

```python
def test_api_with_empty_prefix():
    parser = Parser()
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
    parser.api("root", node, prefix="")
    assert not parser.api.__code__.co_varnames[1] == "prefix"


# LLM-generated content at query #15
#--------------------------

```python
def test_isinstance_node_Import_ImportFrom():
    p = Parser()
    script = "import os\nfrom sys import path"
    p.parse("test_module", script)
    root_node = parse(script, type_comments=True)
    for node in walk_body(root_node.body):
        assert isinstance(node, (Import, ImportFrom))


# LLM-generated content at query #16
#--------------------------

```python
def test__attr_returns_none_when_intermediate_attribute_is_none():
    class MockObject:
        pass

    obj = MockObject()
    obj.a = None
    assert _attr(obj, 'a.b') is None


# LLM-generated content at query #17
#--------------------------

```python
def test_parser_constructor_default():
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

def test_parser_constructor_with_params():
    p = Parser(False, 2, True)
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


# LLM-generated content at query #18
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments(
        posonlyargs=[arg('a', None), arg('b', None)],
        args=[arg('c', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| a | b | / | c | return |\n"
        "|:---:|:---:|:---:|:---:|:---:|\n"
        "| ANY | ANY |  | ANY | ANY |\n\n"
    )

def test_func_api_with_keyword_args():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[arg('a', None), arg('b', None)],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| * | a | b | return |\n"
        "|:---:|:---:|:---:|:---:|\n"
        "|  | ANY | ANY | ANY |\n\n"
    )

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=arg('args', None)
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| a | *args | return |\n"
        "|:---:|:---:|:---:|\n"
        "| ANY |  | ANY |\n\n"
    )

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg('kwargs', None),
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| **kwargs | return |\n"
        "|:---:|:---:|\n"
        "|  | ANY |\n\n"
    )

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None), arg('b', None)],
        defaults=[Constant(value=1), Constant(value=2)],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| a | b | return |\n"
        "|:---:|:---:|:---:|\n"
        "| ANY | ANY | ANY |\n"
        "| 1 | 2 |  |\n\n"
    )

def test_func_api_with_self():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('self', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=False)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| Self | a | return |\n"
        "|:---:|:---:|:---:|\n"
        "| ANY | ANY | ANY |\n\n"
    )

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('cls', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    assert parser.doc['root.func'] == (
        "### func()\n\n*Full name:* `root.func`\n<a id=\"root-func\"></a>\n\n"
        "| type[Self] | a | return |\n"
        "|:---:|:---:|:---:|\n"
        "| ANY | ANY | ANY |\n\n"
    )


# LLM-generated content at query #19
#--------------------------

```python
def test_annassign_name_isinstance():
    node = AnnAssign(target=Name(id='test'), annotation=None, value=None)
    assert isinstance(node, AnnAssign)
    assert isinstance(node.target, Name)


# LLM-generated content at query #20
#--------------------------

```python
def test_is_public_predicate_false():
    p = Parser()
    p.imp = {'root': {'child'}}
    p.root = {'root.child': 'root'}
    assert not p.is_public('root.child')


# LLM-generated content at query #21
#--------------------------

```python
def test_attr_single_level():
    class TestObj:
        pass
    obj = TestObj()
    obj.value = 42
    assert _attr(obj, 'value') == 42

def test_attr_nested_levels():
    class Inner:
        pass
    class Outer:
        pass
    inner = Inner()
    inner.data = "nested"
    outer = Outer()
    outer.inner = inner
    assert _attr(outer, 'inner.data') == "nested"

def test_attr_nonexistent_attribute():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, 'nonexistent') is None

def test_attr_intermediate_none():
    class Inner:
        pass
    class Outer:
        pass
    outer = Outer()
    outer.inner = None
    assert _attr(outer, 'inner.data') is None

def test_attr_empty_string():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, '') is obj


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root = {'pkg': 'pkg'}
    parser.alias = {'pkg.Base': 'base.Base'}
    parser.level = {'pkg': 0}
    parser.doc = {'pkg': '# Module `pkg`\n\n'}

    class_def = ClassDef(
        name='Child',
        bases=[Name(id='Base', ctx=Load())],
        body=[]
    )
    parser.class_api('pkg', 'pkg.Child', class_def.bases, class_def.body)

    assert 'pkg.Child' in parser.doc
    assert 'Bases' in parser.doc['pkg.Child']
    assert 'base.Base' in parser.doc['pkg.Child']

def test_class_api_with_enum():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root = {'pkg': 'pkg'}
    parser.level = {'pkg': 0}
    parser.doc = {'pkg': '# Module `pkg`\n\n'}

    class_def = ClassDef(
        name='Color',
        bases=[Name(id='Enum', ctx=Load())],
        body=[
            AnnAssign(
                target=Name(id='RED', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=None
            )
        ]
    )
    parser.class_api('pkg', 'pkg.Color', class_def.bases, class_def.body)

    assert 'Enums' in parser.doc['pkg.Color']
    assert 'RED' in parser.doc['pkg.Color']

def test_class_api_with_members():
    parser = Parser.new(link=False, level=1, toc=False)
    parser.root = {'pkg': 'pkg'}
    parser.level = {'pkg': 0}
    parser.doc = {'pkg': '# Module `pkg`\n\n'}

    class_def = ClassDef(
        name='MyClass',
        bases=[],
        body=[
            AnnAssign(
                target=Name(id='attr1', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=None
            ),
            Assign(
                targets=[Name(id='attr2', ctx=Store())],
                value=Constant(value=42)
            )
        ]
    )
    parser.class_api('pkg', 'pkg.MyClass', class_def.bases, class_def.body)

    assert 'Members' in parser.doc['pkg.MyClass']
    assert 'attr1' in parser.doc['pkg.MyClass']
    assert 'attr2' in parser.doc['pkg.MyClass']


# LLM-generated content at query #24
#--------------------------

```python
def test_parser_constructor_defaults():
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

def test_parser_constructor_with_custom_values():
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
    assert parser.link == False
    assert parser.b_level == 3
    assert parser.toc == True
    assert parser.level == {}
    assert parser.doc == {}
    assert parser.docstring == {}
    assert parser.imp == {}
    assert parser.root == {}
    assert parser.alias == {}
    assert parser.const == {}

def test_parser_post_init_with_toc_true():
    parser = Parser(toc=True)
    assert parser.link == True
    assert parser.toc == True

def test_parser_post_init_with_toc_false():
    parser = Parser(toc=False)
    assert parser.link == True
    assert parser.toc == False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_globals_with_annassign():
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

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="y")],
        value=Constant(value="hello")
    )
    parser.globals("root", node)
    assert parser.alias["root.y"] == "'hello'"
    assert parser.const["root.y"] == "str"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")])
    )
    parser.globals("root", node)
    assert parser.imp["root"] == {"root.func1", "root.func2"}

def test_globals_ignores_complex_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="x"), Name(id="y")],
        value=Constant(value=1)
    )
    parser.globals("root", node)
    assert "root.x" not in parser.alias
    assert "root.y" not in parser.alias

def test_globals_ignores_non_constant_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=Name(id="some_list")
    )
    parser.globals("root", node)
    assert parser.imp["root"] == set()


# LLM-generated content at query #2
#--------------------------

```python
def test_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="attr1", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=None),
        AnnAssign(target=Name(id="attr2", ctx=Store()), annotation=Name(id="str", ctx=Load()), value=None),
        Assign(targets=[Name(id="attr3", ctx=Store())], value=Constant(value=42)),
        Delete(targets=[Name(id="attr2", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert parser.doc[name] == "# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id=\"test-module-testclass\"></a>\n\n| Bases |\n|:-----:|\n|<code>BaseClass</code>|\n\n| Members | Type |\n|:--------:|:-----:|\n|<code>attr1</code>|<code>int</code>|\n|<code>attr3</code>|<code>int</code>|\n\n"


# LLM-generated content at query #3
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver("root", {})
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_syntax():
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


# LLM-generated content at query #4
#--------------------------

```python
def test_node_type_comment_is_not_none():
    parser = Parser()
    node = Assign(
        targets=[Name(id='x')],
        value=Constant(value=42),
        type_comment='int'
    )
    assert parser.globals('root', node) is None
    assert 'root.x' in parser.alias
    assert parser.alias['root.x'] == '42'
    assert parser.const.get('root.x') == 'int'


# LLM-generated content at query #5
#--------------------------

```python
def test_load_docstring():
    p = Parser()
    p.doc = {'pkg.module': '', 'pkg.module.func': ''}
    p.docstring = {}
    p.root = {'pkg.module': 'pkg.module', 'pkg.module.func': 'pkg.module'}
    m = ModuleType('pkg.module')
    m.func = lambda: None
    m.func.__doc__ = "This is a function."
    p.load_docstring('pkg.module', m)
    assert p.docstring['pkg.module.func'] == "```python\nThis is a function.\n```"


# LLM-generated content at query #6
#--------------------------

```python
def test_api_function():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert parser.doc["test_module.test_func"].startswith("### test_func()")

def test_api_async_function():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test_module.test_async_func" in parser.doc
    assert parser.doc["test_module.test_async_func"].startswith("### async test_async_func()")

def test_api_class():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert parser.doc["test_module.TestClass"].startswith("### class TestClass")

def test_api_with_prefix():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="method", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node, prefix="TestClass")
    assert "test_module.TestClass.method" in parser.doc
    assert parser.doc["test_module.TestClass.method"].startswith("#### method()")

def test_api_with_decorators():
    parser = Parser()
    root = "test_module"
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="decorated_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[decorator])
    parser.api(root, node)
    assert "test_module.decorated_func" in parser.doc
    assert "Decorators" in parser.doc["test_module.decorated_func"]


# LLM-generated content at query #7
#--------------------------

```python
def test_attr_simple_attribute():
    class TestObj:
        pass
    obj = TestObj()
    obj.simple = "value"
    assert _attr(obj, "simple") == "value"

def test_attr_nested_attribute():
    class TestObj:
        pass
    obj = TestObj()
    obj.nested = TestObj()
    obj.nested.attr = "nested_value"
    assert _attr(obj, "nested.attr") == "nested_value"

def test_attr_nonexistent_attribute():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, "nonexistent") is None

def test_attr_chain_break():
    class TestObj:
        pass
    obj = TestObj()
    obj.level1 = TestObj()
    assert _attr(obj, "level1.level2.attr") is None

def test_attr_empty_string():
    class TestObj:
        pass
    obj = TestObj()
    assert _attr(obj, "") == obj

def test_attr_single_dot():
    class TestObj:
        pass
    obj = TestObj()
    obj.a = TestObj()
    obj.a.b = "value"
    assert _attr(obj, ".a.b") is None

def test_attr_trailing_dot():
    class TestObj:
        pass
    obj = TestObj()
    obj.a = "value"
    assert _attr(obj, "a.") is None


# LLM-generated content at query #8
#--------------------------

```python
def test_class_api_annassign_with_name_target():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="test_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=Constant(value=42),
            simple=1
        )
    ]
    parser.class_api(root, name, bases, body)
    assert parser.doc[name] == "#" * (parser.b_level + 2) + " class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"


# LLM-generated content at query #9
#--------------------------

```python
def test_empty_string():
    assert doctest("") == ""

def test_single_line_no_doctest():
    assert doctest("This is a regular line.") == "This is a regular line."

def test_single_line_with_doctest():
    assert doctest(">>> print('hello')") == "```python\n>>> print('hello')\n```"

def test_multiple_lines_no_doctest():
    assert doctest("Line 1\nLine 2\nLine 3") == "Line 1\nLine 2\nLine 3"

def test_multiple_lines_with_single_doctest():
    input_doc = "Line 1\n>>> print('hello')\nLine 2"
    expected = "Line 1\n```python\n>>> print('hello')\n```\nLine 2"
    assert doctest(input_doc) == expected

def test_multiple_lines_with_multiple_doctests():
    input_doc = "Line 1\n>>> print('hello')\nLine 2\n>>> x = 1\nLine 3"
    expected = "Line 1\n```python\n>>> print('hello')\n```\nLine 2\n```python\n>>> x = 1\n```\nLine 3"
    assert doctest(input_doc) == expected

def test_doctest_at_end():
    input_doc = "Line 1\n>>> print('hello')"
    expected = "Line 1\n```python\n>>> print('hello')\n```"
    assert doctest(input_doc) == expected

def test_doctest_at_start():
    input_doc = ">>> print('hello')\nLine 1"
    expected = "```python\n>>> print('hello')\n```\nLine 1"
    assert doctest(input_doc) == expected

def test_consecutive_doctests():
    input_doc = ">>> print('hello')\n>>> x = 1"
    expected = "```python\n>>> print('hello')\n>>> x = 1\n```"
    assert doctest(input_doc) == expected

def test_mixed_content():
    input_doc = ">>> print('hello')\nNormal line\n>>> x = 1\nAnother normal line"
    expected = "```python\n>>> print('hello')\n```\nNormal line\n```python\n>>> x = 1\n```\nAnother normal line"
    assert doctest(input_doc) == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.doc = {'root.public_name': ''}
    parser.root = {'root.public_name': 'root'}
    assert parser.is_public('root.public_name') == True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root._private_name': ''}
    parser.root = {'root._private_name': 'root'}
    assert parser.is_public('root._private_name') == False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root.__magic__': ''}
    parser.root = {'root.__magic__': 'root'}
    assert parser.is_public('root.__magic__') == True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {'root': {'nested'}}
    parser.doc = {'root.nested.public_name': ''}
    parser.root = {'root.nested.public_name': 'root.nested'}
    assert parser.is_public('root.nested.public_name') == True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.doc = {'root.nested._private_name': ''}
    parser.root = {'root.nested._private_name': 'root.nested'}
    assert parser.is_public('root.nested._private_name') == False

def test_is_public_with_listed_in_all():
    parser = Parser()
    parser.imp = {'root': {'listed_name'}}
    parser.doc = {'root.listed_name': ''}
    parser.root = {'root.listed_name': 'root'}
    assert parser.is_public('root.listed_name') == True

def test_is_public_with_parent_listed_in_all():
    parser = Parser()
    parser.imp = {'root': {'parent'}}
    parser.doc = {'root.parent.child': ''}
    parser.root = {'root.parent.child': 'root.parent'}
    assert parser.is_public('root.parent.child') == True

def test_is_public_with_const():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.const = {'root.CONST': 'int'}
    parser.root = {'root.CONST': 'root'}
    assert parser.is_public('root.CONST') == True


# LLM-generated content at query #11
#--------------------------

```python
def test_visit_Attribute_removes_typing_prefix():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "List"
    assert isinstance(result.ctx, Load)

def test_visit_Attribute_keeps_non_typing_attribute():
    resolver = Resolver(root="", alias={})
    node = Attribute(value=Name(id="other", ctx=Load()), attr="List", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Attribute)
    assert result.value.id == "other"
    assert result.attr == "List"
    assert isinstance(result.ctx, Load)


# LLM-generated content at query #12
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
    parser.parse("test_module", script)
    assert "test_module" in parser.doc
    assert "test_module.foo" in parser.doc
    assert "test_module.x" in parser.alias

def test_parse_with_imports():
    parser = Parser()
    script = """
    import os
    from sys import path
    x = 1
    """
    parser.parse("test_module", script)
    assert "os" in parser.alias
    assert "test_module.path" in parser.alias

def test_parse_with_annotations():
    parser = Parser()
    script = """
    x: int = 1
    y: str = "hello"
    """
    parser.parse("test_module", script)
    assert "test_module.x" in parser.alias
    assert "test_module.y" in parser.alias

def test_parse_with_classes():
    parser = Parser()
    script = """
    class MyClass:
        \"\"\"Class docstring.\"\"\"
        def method(self):
            pass
    """
    parser.parse("test_module", script)
    assert "test_module.MyClass" in parser.doc
    assert "test_module.MyClass.method" in parser.doc

def test_parse_with_decorators():
    parser = Parser()
    script = """
    @decorator
    def foo():
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.doc
    assert "Decorator" in parser.doc["test_module.foo"]

def test_parse_with_constants():
    parser = Parser()
    script = """
    CONSTANT = 42
    """
    parser.parse("test_module", script)
    assert "test_module.CONSTANT" in parser.const

def test_parse_with_all():
    parser = Parser()
    script = """
    __all__ = ["foo", "bar"]
    def foo():
        pass
    def bar():
        pass
    def baz():
        pass
    """
    parser.parse("test_module", script)
    assert "test_module.foo" in parser.imp["test_module"]
    assert "test_module.bar" in parser.imp["test_module"]
    assert "test_module.baz" not in parser.imp["test_module"]


# LLM-generated content at query #13
#--------------------------

```python
def test_is_public_returns_false_when_all_l_is_empty():
    parser = Parser()
    parser.imp = {}
    parser.root = {}
    assert not parser.is_public("test_name")


# LLM-generated content at query #14
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

def test_class_api_with_enum_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Name(id="Enum", ctx=Load())]
    body = [
        AnnAssign(target=Name(id="MEMBER1", ctx=Store()), annotation=Constant(value=1), value=Constant(value=1)),
        AnnAssign(target=Name(id="MEMBER2", ctx=Store()), annotation=Constant(value=2), value=Constant(value=2))
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "MEMBER1" in parser.doc[name]
    assert "MEMBER2" in parser.doc[name]

def test_class_api_with_public_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_member", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=42)),
        Assign(targets=[Name(id="another_member", ctx=Store())], value=Constant(value="hello"))
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "public_member" in parser.doc[name]
    assert "another_member" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        AnnAssign(target=Name(id="public_member", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=42)),
        Delete(targets=[Name(id="public_member", ctx=Del())])
    ]
    parser.class_api(root, name, bases, body)
    assert "public_member" not in parser.doc[name]


# LLM-generated content at query #15
#--------------------------

```python
def test__attr_returns_none_when_attribute_not_found():
    class Dummy:
        pass

    obj = Dummy()
    result = _attr(obj, 'nonexistent')
    assert result is None


# LLM-generated content at query #16
#--------------------------

```python
def test_imports_with_Import_node():
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"

def test_imports_with_Import_node_and_asname():
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os", asname="operating_system")])
    parser.imports(root, node)
    assert parser.alias["test_module.operating_system"] == "os"

def test_imports_with_ImportFrom_node_and_level():
    parser = Parser()
    root = "test_module.sub_module"
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.sub_module.path"] == "test_module.os.path"

def test_imports_with_ImportFrom_node_and_asname():
    parser = Parser()
    root = "test_module"
    node = ImportFrom(module="os", names=[alias(name="path", asname="os_path")], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.os_path"] == "os.path"


# LLM-generated content at query #17
#--------------------------

```python
def test_compile_empty():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == "\n"

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc["root"] = "# Module `{}`"
    p.level["root"] = 0
    p.root["root"] = "root"
    assert p.compile() == "**Table of contents:**\n\n# Module `root`\n"

def test_compile_with_magic_name():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["__init__"] = "## `__init__()`\n\n*Full name:* `{}`\n\n"
    p.level["__init__"] = 1
    p.root["__init__"] = "root"
    assert p.compile() == "## `__init__()`\n\n*Full name:* `__init__`\n\n"

def test_compile_with_docstring():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root.func"] = "### `func()`\n\n*Full name:* `{}`\n\n"
    p.level["root.func"] = 1
    p.root["root.func"] = "root"
    p.docstring["root.func"] = "This is a function."
    assert p.compile() == "### `func()`\n\n*Full name:* `root.func`\n\nThis is a function."

def test_compile_with_const():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root"] = "# Module `{}`"
    p.level["root"] = 0
    p.root["root"] = "root"
    p.const["root.CONST"] = "int"
    p.imp["root"] = {"CONST"}
    assert p.compile() == "# Module `root`\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n"

def test_compile_with_non_public():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root._private"] = "### `_private()`\n\n*Full name:* `{}`\n\n"
    p.level["root._private"] = 1
    p.root["root._private"] = "root"
    assert p.compile() == "\n"


# LLM-generated content at query #18
#--------------------------

```python
def test_defaults_with_none_and_non_none_values():
    args = [None, "x", None, "y"]
    result = list(_defaults(args))
    assert result == [" ", "`x`", " ", "`y`"]

def test_defaults_with_empty_string():
    args = ["", None]
    result = list(_defaults(args))
    assert result == [" ", " "]

def test_defaults_with_special_characters():
    args = ["a&b", "c|d"]
    result = list(_defaults(args))
    assert result == ["<code>a&b</code>", "`c&#124;d`"]

def test_defaults_with_all_none():
    args = [None, None, None]
    result = list(_defaults(args))
    assert result == [" ", " ", " "]

def test_defaults_with_empty_list():
    args = []
    result = list(_defaults(args))
    assert result == []


# LLM-generated content at query #19
#--------------------------

```python
def test_class_api_with_bases():
    parser = Parser()
    bases = [Name(id='BaseClass', ctx=Load())]
    body = []
    parser.class_api('root', 'root.ChildClass', bases, body)
    assert '| Bases |\n|:---:|\n| `BaseClass` |\n\n' in parser.doc['root.ChildClass']

def test_class_api_without_bases():
    parser = Parser()
    bases = []
    body = []
    parser.class_api('root', 'root.ChildClass', bases, body)
    assert 'Bases' not in parser.doc['root.ChildClass']

def test_class_api_with_enum():
    parser = Parser()
    bases = [Name(id='Enum', ctx=Load())]
    body = [
        AnnAssign(target=Name(id='MEMBER1', ctx=Store()), annotation=Constant(value=1), value=Constant(value=1)),
        AnnAssign(target=Name(id='MEMBER2', ctx=Store()), annotation=Constant(value=2), value=Constant(value=2))
    ]
    parser.class_api('root', 'root.MyEnum', bases, body)
    assert '| Enums |\n|:---:|\n| `MEMBER1` |\n| `MEMBER2` |\n\n' in parser.doc['root.MyEnum']

def test_class_api_with_members():
    parser = Parser()
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None),
        AnnAssign(target=Name(id='attr2', ctx=Store()), annotation=Name(id='str', ctx=Load()), value=None)
    ]
    parser.class_api('root', 'root.MyClass', bases, body)
    assert '| Members | Type |\n|:---:|:---:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n' in parser.doc['root.MyClass']

def test_class_api_with_deleted_members():
    parser = Parser()
    bases = []
    body = [
        AnnAssign(target=Name(id='attr1', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None),
        Delete(targets=[Name(id='attr1', ctx=Del())])
    ]
    parser.class_api('root', 'root.MyClass', bases, body)
    assert 'Members' not in parser.doc['root.MyClass']

def test_class_api_with_private_members():
    parser = Parser()
    bases = []
    body = [
        AnnAssign(target=Name(id='_private', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=None)
    ]
    parser.class_api('root', 'root.MyClass', bases, body)
    assert 'Members' not in parser.doc['root.MyClass']


# LLM-generated content at query #20
#--------------------------

```python
def test_is_enum_when_assign_node_and_enum_true():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("enum.Enum").body[0].value]
    body = [Assign(targets=[Name(id="VALUE")], value=Constant(value=1))]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]


# LLM-generated content at query #21
#--------------------------

```python
def test_defaults_with_non_none_args():
    args = [expr("x"), expr("y"), expr("z")]
    result = list(_defaults(args))
    assert result == [code(unparse(expr("x"))), code(unparse(expr("y"))), code(unparse(expr("z")))]


# LLM-generated content at query #22
#--------------------------

```python
def test_walk_body_empty_sequence():
    assert list(walk_body([])) == []

def test_walk_body_single_statement():
    stmt = Assign(targets=[Name(id='x')], value=Constant(value=1))
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_if_statement():
    if_node = If(test=Name(id='x'), body=[Assign(targets=[Name(id='y')], value=Constant(value=2))], orelse=[])
    assert list(walk_body([if_node])) == [Assign(targets=[Name(id='y')], value=Constant(value=2))]

def test_walk_body_if_with_orelse():
    if_node = If(
        test=Name(id='x'),
        body=[Assign(targets=[Name(id='y')], value=Constant(value=2))],
        orelse=[Assign(targets=[Name(id='z')], value=Constant(value=3))]
    )
    assert list(walk_body([if_node])) == [
        Assign(targets=[Name(id='y')], value=Constant(value=2)),
        Assign(targets=[Name(id='z')], value=Constant(value=3))
    ]

def test_walk_body_try_statement():
    try_node = Try(
        body=[Assign(targets=[Name(id='a')], value=Constant(value=1))],
        handlers=[],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [Assign(targets=[Name(id='a')], value=Constant(value=1))]

def test_walk_body_try_with_handler():
    try_node = Try(
        body=[Assign(targets=[Name(id='a')], value=Constant(value=1))],
        handlers=[ExceptHandler(body=[Assign(targets=[Name(id='b')], value=Constant(value=2))])],
        orelse=[],
        finalbody=[]
    )
    assert list(walk_body([try_node])) == [
        Assign(targets=[Name(id='a')], value=Constant(value=1)),
        Assign(targets=[Name(id='b')], value=Constant(value=2))
    ]

def test_walk_body_try_with_orelse_and_finalbody():
    try_node = Try(
        body=[Assign(targets=[Name(id='a')], value=Constant(value=1))],
        handlers=[],
        orelse=[Assign(targets=[Name(id='b')], value=Constant(value=2))],
        finalbody=[Assign(targets=[Name(id='c')], value=Constant(value=3))]
    )
    assert list(walk_body([try_node])) == [
        Assign(targets=[Name(id='a')], value=Constant(value=1)),
        Assign(targets=[Name(id='b')], value=Constant(value=2)),
        Assign(targets=[Name(id='c')], value=Constant(value=3))
    ]

def test_walk_body_mixed_statements():
    body = [
        Assign(targets=[Name(id='x')], value=Constant(value=1)),
        If(
            test=Name(id='x'),
            body=[Assign(targets=[Name(id='y')], value=Constant(value=2))],
            orelse=[Assign(targets=[Name(id='z')], value=Constant(value=3))]
        ),
        Try(
            body=[Assign(targets=[Name(id='a')], value=Constant(value=4))],
            handlers=[ExceptHandler(body=[Assign(targets=[Name(id='b')], value=Constant(value=5))])],
            orelse=[Assign(targets=[Name(id='c')], value=Constant(value=6))],
            finalbody=[Assign(targets=[Name(id='d')], value=Constant(value=7))]
        )
    ]
    assert list(walk_body(body)) == [
        Assign(targets=[Name(id='x')], value=Constant(value=1)),
        Assign(targets=[Name(id='y')], value=Constant(value=2)),
        Assign(targets=[Name(id='z')], value=Constant(value=3)),
        Assign(targets=[Name(id='a')], value=Constant(value=4)),
        Assign(targets=[Name(id='b')], value=Constant(value=5)),
        Assign(targets=[Name(id='c')], value=Constant(value=6)),
        Assign(targets=[Name(id='d')], value=Constant(value=7))
    ]


# LLM-generated content at query #23
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments(posonlyargs=[arg('a', None), arg('b', None)], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_keyword_args():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[arg('a', None), arg('b', None)], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `Any` | `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=arg('args', None))
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| *args | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), vararg=None)
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| **kwargs | return |\n|:---:|:---:|\n| `Any` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('a', None), arg('b', None)], defaults=[Constant(1), Constant(2)], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    expected = "| a | b | return |\n|:---:|:---:|:---:|\n| `Any` | `Any` | `Any` |\n| `1` | `2` |  |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_self():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('self', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=False)
    expected = "| self | return |\n|:---:|:---:|\n| `Self` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[arg('cls', None)], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    expected = "| cls | return |\n|:---:|:---:|\n| `type[Self]` | `Any` |\n\n"
    assert parser.doc['root.func'] == expected

def test_func_api_with_return_annotation():
    parser = Parser()
    node = arguments(posonlyargs=[], args=[], defaults=[], kwonlyargs=[], kw_defaults=[], kwarg=None, vararg=None)
    parser.func_api('root', 'func', node, Constant('int'), has_self=False, cls_method=False)
    expected = "| return |\n|:---:|\n| `int` |\n\n"
    assert parser.doc['root.func'] == expected


# LLM-generated content at query #24
#--------------------------

```python
def test_class_api_with_empty_bases():
    parser = Parser()
    parser.class_api("root", "class_name", [], [])
    assert "Bases" not in parser.doc["root.class_name"]


# LLM-generated content at query #25
#--------------------------

```python
def test_load_docstring_updates_docstring_when_doc_exists():
    parser = Parser()
    parser.doc = {'pkg.submod': '# Module `pkg.submod`'}
    parser.docstring = {}
    mock_module = type('MockModule', (), {'submod': {'__doc__': 'Test doc'}})()
    parser.load_docstring('pkg', mock_module)
    assert 'pkg.submod' in parser.docstring
    assert parser.docstring['pkg.submod'] == 'Test doc'


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_evaluates_to_false():
    p = Parser()
    node = Assign(targets=[Name(id='x')], value=Constant(value=123))
    p.globals('root', node)
    assert '__all__' not in p.imp


# LLM-generated content at query #27
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    node = AnnAssign(
        target=Name(id="var"),
        annotation=Name(id="int"),
        value=Constant(value=1)
    )
    parser.globals("module", node)
    assert parser.alias["module.var"] == "1"
    assert parser.const["module.VAR"] == "int"
    assert parser.root["module.VAR"] == "module"

def test_globals_with_assign():
    parser = Parser()
    node = Assign(
        targets=[Name(id="var")],
        value=Constant(value=1)
    )
    parser.globals("module", node)
    assert parser.alias["module.var"] == "1"
    assert parser.const["module.VAR"] == "int"

def test_globals_with_all():
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__")],
        value=List(elts=[Constant(value="func")])
    )
    parser.globals("module", node)
    assert "module.func" in parser.imp["module"]


# LLM-generated content at query #2
#--------------------------

```python
def test_is_public_family_with_public_name():
    assert is_public_family('module.submodule.function') == True

def test_is_public_family_with_private_name():
    assert is_public_family('module._private_function') == False

def test_is_public_family_with_local_name():
    assert is_public_family('_local_function') == False

def test_is_public_family_with_magic_name():
    assert is_public_family('module.__init__') == True

def test_is_public_family_with_mixed_names():
    assert is_public_family('module._private.__magic__') == False

def test_is_public_family_with_empty_name():
    assert is_public_family('') == True

def test_is_public_family_with_single_public_name():
    assert is_public_family('public') == True

def test_is_public_family_with_single_private_name():
    assert is_public_family('_private') == False

def test_is_public_family_with_single_magic_name():
    assert is_public_family('__magic__') == True


# LLM-generated content at query #3
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

def test_imports_with_import_from_node_and_level():
    parser = Parser()
    root = "test.module.submodule"
    node = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, node)
    assert parser.alias["test.module.submodule.path"] == "test.module.os.path"

def test_imports_with_import_from_node_and_asname():
    parser = Parser()
    root = "test.module"
    node = ImportFrom(module="os.path", names=[alias(name="join", asname="j")], level=0)
    parser.imports(root, node)
    assert parser.alias["test.module.j"] == "os.path.join"


# LLM-generated content at query #4
#--------------------------

```python
def test_imports_with_import_node():
    p = Parser()
    script = "import os"
    root_node = parse(script, type_comments=True)
    node = root_node.body[0]
    assert isinstance(node, Import)
    p.imports("test", node)
    assert p.alias["test.os"] == "os"


# LLM-generated content at query #5
#--------------------------

```python
def test_visit_Constant_with_non_string_value():
    resolver = Resolver(root="test", alias={})
    node = Constant(value=123)
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_invalid_syntax():
    resolver = Resolver(root="test", alias={})
    node = Constant(value="invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node

def test_visit_Constant_with_valid_name():
    resolver = Resolver(root="test", alias={"test.name": "test.alias"})
    node = Constant(value="name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias"


# LLM-generated content at query #6
#--------------------------

```python
def test_table_with_single_column():
    result = table('Title', items=['Item1', 'Item2'])
    assert result == "| Title |\n|:-----:|\n| Item1 |\n| Item2 |\n\n"

def test_table_with_multiple_columns():
    result = table('A', 'B', items=[['1', '2'], ['3', '4']])
    assert result == "| A | B |\n|:---:|:---:|\n| 1 | 2 |\n| 3 | 4 |\n\n"

def test_table_with_mixed_items():
    result = table('X', 'Y', items=['Single', ['A', 'B']])
    assert result == "| X | Y |\n|:---:|:---:|\n| Single |   |\n| A | B |\n\n"

def test_table_with_empty_titles():
    result = table('', items=[['A', 'B']])
    assert result == "|   |   |\n|:---:|:---:|\n| A | B |\n\n"

def test_table_with_long_titles():
    result = table('VeryLongTitle', 'AnotherLongTitle', items=[['Short', 'Data']])
    assert result == "| VeryLongTitle | AnotherLongTitle |\n|:--------------:|:----------------:|\n| Short | Data |\n\n"


# LLM-generated content at query #7
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass", mode='eval').body]
    body = [
        parse("x: int = 1", mode='eval').body,
        parse("y: str = 'hello'", mode='eval').body,
        parse("z = 3.14", mode='eval').body,
    ]
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "Type" in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum", mode='eval').body]
    body = [
        parse("A = 1", mode='eval').body,
        parse("B = 2", mode='eval').body,
    ]
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "A" in parser.doc[name]
    assert "B" in parser.doc[name]

def test_class_api_with_no_bases_or_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.EmptyClass"
    bases = []
    body = []
    parser.class_api(root, name, bases, body)
    assert "Bases" not in parser.doc[name]
    assert "Members" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("_private: int = 1", mode='eval').body,
        parse("public: str = 'hello'", mode='eval').body,
    ]
    parser.class_api(root, name, bases, body)
    assert "Members" in parser.doc[name]
    assert "_private" not in parser.doc[name]
    assert "public" in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("x: int = 1", mode='eval').body,
        parse("del x", mode='eval').body,
    ]
    parser.class_api(root, name, bases, body)
    assert "x" not in parser.doc[name]


# LLM-generated content at query #8
#--------------------------

```python
def test_class_api_with_delete_node():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        Delete(targets=[Name(id="attr1")]),
        Delete(targets=[Name(id="attr2")])
    ]
    parser.class_api(root, name, bases, body)
    assert "attr1" not in parser.doc[name]
    assert "attr2" not in parser.doc[name]


# LLM-generated content at query #9
#--------------------------

```python
def test_doctest_empty_string():
    assert doctest("") == ""

def test_doctest_single_line_no_doctest():
    assert doctest("This is a regular line.") == "This is a regular line."

def test_doctest_single_line_with_doctest():
    assert doctest(">>> print('hello')") == "```python\n>>> print('hello')\n```"

def test_doctest_multiple_lines_no_doctest():
    assert doctest("Line 1\nLine 2\nLine 3") == "Line 1\nLine 2\nLine 3"

def test_doctest_multiple_lines_with_single_doctest():
    assert doctest("Regular line\n>>> print('hello')\nAnother regular line") == "Regular line\n```python\n>>> print('hello')\n```\nAnother regular line"

def test_doctest_multiple_lines_with_multiple_doctests():
    assert doctest("Regular line\n>>> print('hello')\n>>> print('world')\nAnother regular line") == "Regular line\n```python\n>>> print('hello')\n>>> print('world')\n```\nAnother regular line"

def test_doctest_multiple_lines_with_doctest_at_end():
    assert doctest("Regular line\n>>> print('hello')") == "Regular line\n```python\n>>> print('hello')\n```"

def test_doctest_multiple_lines_with_doctest_at_start():
    assert doctest(">>> print('hello')\nRegular line") == "```python\n>>> print('hello')\n```\nRegular line"

def test_doctest_multiple_lines_with_doctest_in_middle():
    assert doctest("Regular line 1\n>>> print('hello')\nRegular line 2") == "Regular line 1\n```python\n>>> print('hello')\n```\nRegular line 2"


# LLM-generated content at query #10
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver(root="root", alias={}, self_ty="T")
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver(root="root", alias={"root.Name": "alias.Name"})
    node = Name(id="Name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "alias.Name"

def test_visit_Name_without_alias():
    resolver = Resolver(root="root", alias={})
    node = Name(id="Name", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Name"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver(root="root", alias={"root.T": "typing.TypeVar('T')"})
    node = Name(id="T", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "T"


# LLM-generated content at query #11
#--------------------------

```python
def test_const_type_constant():
    assert const_type(Constant(5)) == "int"
    assert const_type(Constant("hello")) == "str"
    assert const_type(Constant(3.14)) == "float"
    assert const_type(Constant(True)) == "bool"

def test_const_type_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant("a"), Constant("b")])) == "tuple[str, str]"
    assert const_type(Tuple([Constant(1), Constant("a")])) == "tuple[Any, Any]"

def test_const_type_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int]"
    assert const_type(List([Constant("a"), Constant("b")])) == "list[str]"
    assert const_type(List([Constant(1), Constant("a")])) == "list[Any]"

def test_const_type_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int]"
    assert const_type(Set([Constant("a"), Constant("b")])) == "set[str]"
    assert const_type(Set([Constant(1), Constant("a")])) == "set[Any]"

def test_const_type_dict():
    assert const_type(Dict([Constant(1)], [Constant("a")])) == "dict[int, str]"
    assert const_type(Dict([Constant("a")], [Constant(1)])) == "dict[str, int]"
    assert const_type(Dict([Constant(1), Constant("a")], [Constant("b"), Constant(2)])) == "dict[Any, Any]"

def test_const_type_call():
    assert const_type(Call(Name("int"), [])) == "int"
    assert const_type(Call(Name("str"), [])) == "str"
    assert const_type(Call(Attribute(Value(Constant(1)), "real"), [])) == "float"

def test_const_type_any():
    assert const_type(Call(Name("unknown"), [])) == "Any"
    assert const_type(Name("x")) == "Any"


# LLM-generated content at query #12
#--------------------------

```python
def test_load_docstring_with_valid_module():
    parser = Parser()
    parser.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'Function `func`'}
    parser.docstring = {}
    parser.root = {'pkg.module': 'pkg.module', 'pkg.module.func': 'pkg.module'}
    module = type('module', (), {'__doc__': 'Module doc', 'func': lambda: None})()
    module.func.__doc__ = 'Function doc'
    parser.load_docstring('pkg.module', module)
    assert parser.docstring['pkg.module'] == 'Module doc'
    assert parser.docstring['pkg.module.func'] == 'Function doc'

def test_load_docstring_with_none_doc():
    parser = Parser()
    parser.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.func': 'Function `func`'}
    parser.docstring = {}
    parser.root = {'pkg.module': 'pkg.module', 'pkg.module.func': 'pkg.module'}
    module = type('module', (), {'__doc__': None, 'func': lambda: None})()
    module.func.__doc__ = None
    parser.load_docstring('pkg.module', module)
    assert 'pkg.module' not in parser.docstring
    assert 'pkg.module.func' not in parser.docstring

def test_load_docstring_with_nested_attribute():
    parser = Parser()
    parser.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.Class.attr': 'Attribute `attr`'}
    parser.docstring = {}
    parser.root = {'pkg.module': 'pkg.module', 'pkg.module.Class.attr': 'pkg.module'}
    module = type('module', (), {'__doc__': 'Module doc', 'Class': type('Class', (), {'attr': 'value'})})()
    module.Class.attr.__doc__ = 'Attribute doc'
    parser.load_docstring('pkg.module', module)
    assert parser.docstring['pkg.module'] == 'Module doc'
    assert parser.docstring['pkg.module.Class.attr'] == 'Attribute doc'

def test_load_docstring_with_missing_attribute():
    parser = Parser()
    parser.doc = {'pkg.module': 'Module `pkg.module`', 'pkg.module.missing': 'Missing `missing`'}
    parser.docstring = {}
    parser.root = {'pkg.module': 'pkg.module', 'pkg.module.missing': 'pkg.module'}
    module = type('module', (), {'__doc__': 'Module doc'})()
    parser.load_docstring('pkg.module', module)
    assert parser.docstring['pkg.module'] == 'Module doc'
    assert 'pkg.module.missing' not in parser.docstring


# LLM-generated content at query #13
#--------------------------

```python
def test_is_public_with_all_list():
    p = Parser()
    p.imp = {'pkg': {'mod1', 'mod2'}}
    p.root = {'pkg.mod1': 'pkg', 'pkg.mod2': 'pkg'}
    p.doc = {'pkg.mod1': '', 'pkg.mod2': ''}
    assert p.is_public('pkg.mod1') == True
    assert p.is_public('pkg.mod2') == True

def test_is_public_without_all_list():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg.mod1': 'pkg', 'pkg.mod2': 'pkg'}
    p.doc = {'pkg.mod1': '', 'pkg.mod2': ''}
    assert p.is_public('pkg.mod1') == True
    assert p.is_public('pkg.mod2') == True

def test_is_public_private_name():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg._private': 'pkg'}
    p.doc = {'pkg._private': ''}
    assert p.is_public('pkg._private') == False

def test_is_public_magic_name():
    p = Parser()
    p.imp = {'pkg': set()}
    p.root = {'pkg.__init__': 'pkg'}
    p.doc = {'pkg.__init__': ''}
    assert p.is_public('pkg.__init__') == True

def test_is_public_nested_module():
    p = Parser()
    p.imp = {'pkg': {'mod1'}}
    p.root = {'pkg.mod1.submod': 'pkg'}
    p.doc = {'pkg.mod1.submod': ''}
    assert p.is_public('pkg.mod1.submod') == True

def test_is_public_not_in_all():
    p = Parser()
    p.imp = {'pkg': {'mod1'}}
    p.root = {'pkg.mod2': 'pkg'}
    p.doc = {'pkg.mod2': ''}
    assert p.is_public('pkg.mod2') == False


# LLM-generated content at query #14
#--------------------------

```python
def test_is_public_with_public_name():
    parser = Parser()
    parser.imp = {'root': {'public_name'}}
    parser.root = {'public_name': 'root'}
    parser.doc = {'public_name': ''}
    assert parser.is_public('public_name') is True

def test_is_public_with_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'_private_name': 'root'}
    assert parser.is_public('_private_name') is False

def test_is_public_with_magic_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'__magic__': 'root'}
    parser.doc = {'__magic__': ''}
    assert parser.is_public('__magic__') is True

def test_is_public_with_nested_public_name():
    parser = Parser()
    parser.imp = {'root': {'nested'}}
    parser.root = {'root.nested.public_name': 'root.nested'}
    parser.doc = {'root.nested.public_name': ''}
    assert parser.is_public('root.nested.public_name') is True

def test_is_public_with_nested_private_name():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'root.nested._private_name': 'root.nested'}
    assert parser.is_public('root.nested._private_name') is False

def test_is_public_with_all_listed_name():
    parser = Parser()
    parser.imp = {'root': {'listed_name'}}
    parser.root = {'root.listed_name': 'root'}
    parser.doc = {'root.listed_name': ''}
    assert parser.is_public('root.listed_name') is True

def test_is_public_with_parent_in_all():
    parser = Parser()
    parser.imp = {'root': {'parent'}}
    parser.root = {'root.parent.child': 'root.parent'}
    parser.doc = {'root.parent.child': ''}
    assert parser.is_public('root.parent.child') is True

def test_is_public_with_empty_all():
    parser = Parser()
    parser.imp = {'root': set()}
    parser.root = {'public_name': 'root'}
    parser.doc = {'public_name': ''}
    assert parser.is_public('public_name') is True

def test_is_public_with_no_doc():
    parser = Parser()
    parser.imp = {'root': {'no_doc'}}
    parser.root = {'no_doc': 'root'}
    assert parser.is_public('no_doc') is False


# LLM-generated content at query #15
#--------------------------

```python
def test_const_type_with_constant():
    assert const_type(Constant(1)) == "int"
    assert const_type(Constant(1.0)) == "float"
    assert const_type(Constant("hello")) == "str"
    assert const_type(Constant(True)) == "bool"

def test_const_type_with_tuple():
    assert const_type(Tuple([Constant(1), Constant(2)])) == "tuple[int, int]"
    assert const_type(Tuple([Constant(1), Constant(2.0)])) == "tuple[Any, Any]"
    assert const_type(Tuple([Constant(1), Constant("hello")])) == "tuple[Any, Any]"

def test_const_type_with_list():
    assert const_type(List([Constant(1), Constant(2)])) == "list[int, int]"
    assert const_type(List([Constant(1), Constant(2.0)])) == "list[Any, Any]"
    assert const_type(List([Constant(1), Constant("hello")])) == "list[Any, Any]"

def test_const_type_with_set():
    assert const_type(Set([Constant(1), Constant(2)])) == "set[int, int]"
    assert const_type(Set([Constant(1), Constant(2.0)])) == "set[Any, Any]"
    assert const_type(Set([Constant(1), Constant("hello")])) == "set[Any, Any]"

def test_const_type_with_dict():
    assert const_type(Dict([Constant(1), Constant(2)], [Constant(3), Constant(4)])) == "dict[int, int]"
    assert const_type(Dict([Constant(1), Constant(2.0)], [Constant(3), Constant(4)])) == "dict[Any, Any]"
    assert const_type(Dict([Constant(1), Constant("hello")], [Constant(3), Constant(4)])) == "dict[Any, Any]"

def test_const_type_with_call():
    assert const_type(Call(Name("bool"), [])) == "bool"
    assert const_type(Call(Name("int"), [])) == "int"
    assert const_type(Call(Name("float"), [])) == "float"
    assert const_type(Call(Name("complex"), [])) == "complex"
    assert const_type(Call(Name("str"), [])) == "str"

def test_const_type_with_unsupported_node():
    assert const_type(Name("x")) == "Any"
    assert const_type(Attribute(Name("x"), "y")) == "Any"


# LLM-generated content at query #16
#--------------------------

```python
def test_attr_simple_attribute():
    class TestClass:
        pass
    obj = TestClass()
    obj.simple_attr = "value"
    assert _attr(obj, "simple_attr") == "value"

def test_attr_nested_attribute():
    class InnerClass:
        pass
    class OuterClass:
        pass
    inner = InnerClass()
    inner.nested_attr = "nested_value"
    outer = OuterClass()
    outer.inner = inner
    assert _attr(outer, "inner.nested_attr") == "nested_value"

def test_attr_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "nonexistent_attr") is None

def test_attr_nonexistent_nested_attribute():
    class InnerClass:
        pass
    class OuterClass:
        pass
    inner = InnerClass()
    outer = OuterClass()
    outer.inner = inner
    assert _attr(outer, "inner.nonexistent_attr") is None

def test_attr_middle_nonexistent_attribute():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "nonexistent_attr.nested_attr") is None

def test_attr_empty_string():
    class TestClass:
        pass
    obj = TestClass()
    assert _attr(obj, "") is obj


# LLM-generated content at query #17
#--------------------------

```python
def test_api_function():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "## test_func()" in parser.doc["test_module.test_func"]

def test_api_async_function():
    parser = Parser()
    root = "test_module"
    node = AsyncFunctionDef(name="test_async_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node)
    assert "test_module.test_async_func" in parser.doc
    assert "## async test_async_func()" in parser.doc["test_module.test_async_func"]

def test_api_class():
    parser = Parser()
    root = "test_module"
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    parser.api(root, node)
    assert "test_module.TestClass" in parser.doc
    assert "## class TestClass" in parser.doc["test_module.TestClass"]

def test_api_with_decorators():
    parser = Parser()
    root = "test_module"
    decorator = Name(id="decorator", ctx=Load())
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[decorator])
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "Decorators" in parser.doc["test_module.test_func"]

def test_api_with_docstring():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[], body=[Expr(value=Constant(value="Test docstring"))])
    parser.api(root, node)
    assert "test_module.test_func" in parser.doc
    assert "Test docstring" in parser.docstring["test_module.test_func"]

def test_api_with_prefix():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), returns=None, decorator_list=[])
    parser.api(root, node, prefix="ClassName")
    assert "test_module.ClassName.test_func" in parser.doc
    assert "### test_func()" in parser.doc["test_module.ClassName.test_func"]


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parser = Parser()
    parser.globals("root", Assign(targets=[Name(id="x")], value=Constant(value=1), type_comment="int"))
    assert parser.const.get("root.x", "ANY") != "ANY"


# LLM-generated content at query #19
#--------------------------

```python
def test__defaults_with_none_values():
    args = [None, None]
    result = list(_defaults(args))
    assert result == [" ", " "]

def test__defaults_with_valid_expressions():
    from ast import Name, Constant
    args = [Name(id='x'), Constant(value=42)]
    result = list(_defaults(args))
    assert result == ["`x`", "`42`"]

def test__defaults_with_ampersand_in_expression():
    from ast import BinOp, Name, Add
    args = [BinOp(left=Name(id='a'), op=Add(), right=Name(id='b'))]
    result = list(_defaults(args))
    assert result == ["<code>a + b</code>"]

def test__defaults_with_pipe_in_expression():
    from ast import BinOp, Name, BitOr
    args = [BinOp(left=Name(id='x'), op=BitOr(), right=Name(id='y'))]
    result = list(_defaults(args))
    assert result == ["`x &#124; y`"]

def test__defaults_with_mixed_values():
    from ast import Name, Constant
    args = [None, Name(id='var'), Constant(value=100), None]
    result = list(_defaults(args))
    assert result == [" ", "`var`", "`100`", " "]


# LLM-generated content at query #20
#--------------------------

```python
def test_walk_body_with_empty_sequence():
    assert list(walk_body([])) == []

def test_walk_body_with_single_statement():
    stmt = stmt()
    assert list(walk_body([stmt])) == [stmt]

def test_walk_body_with_if_statement():
    if_node = If([], [])
    assert list(walk_body([if_node])) == []

def test_walk_body_with_if_statement_and_body():
    body_stmt = stmt()
    if_node = If([body_stmt], [])
    assert list(walk_body([if_node])) == [body_stmt]

def test_walk_body_with_if_statement_and_orelse():
    orelse_stmt = stmt()
    if_node = If([], [orelse_stmt])
    assert list(walk_body([if_node])) == [orelse_stmt]

def test_walk_body_with_if_statement_and_body_and_orelse():
    body_stmt = stmt()
    orelse_stmt = stmt()
    if_node = If([body_stmt], [orelse_stmt])
    assert list(walk_body([if_node])) == [body_stmt, orelse_stmt]

def test_walk_body_with_try_statement():
    try_node = Try([], [], [], [], [])
    assert list(walk_body([try_node])) == []

def test_walk_body_with_try_statement_and_body():
    body_stmt = stmt()
    try_node = Try([body_stmt], [], [], [], [])
    assert list(walk_body([try_node])) == [body_stmt]

def test_walk_body_with_try_statement_and_handler():
    handler_stmt = stmt()
    handler = ExceptHandler([], [handler_stmt])
    try_node = Try([], [handler], [], [], [])
    assert list(walk_body([try_node])) == [handler_stmt]

def test_walk_body_with_try_statement_and_orelse():
    orelse_stmt = stmt()
    try_node = Try([], [], [orelse_stmt], [], [])
    assert list(walk_body([try_node])) == [orelse_stmt]

def test_walk_body_with_try_statement_and_finalbody():
    finalbody_stmt = stmt()
    try_node = Try([], [], [], [finalbody_stmt], [])
    assert list(walk_body([try_node])) == [finalbody_stmt]

def test_walk_body_with_mixed_statements():
    stmt1 = stmt()
    if_node = If([stmt1], [])
    stmt2 = stmt()
    try_node = Try([stmt2], [], [], [], [])
    assert list(walk_body([if_node, try_node])) == [stmt1, stmt2]


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_evaluates_to_false():
    obj = object()
    attr = "nonexistent_attribute"
    result = _attr(obj, attr)
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_is_public_returns_false_when_all_l_is_empty():
    parser = Parser()
    parser.root = {"test.module": "test"}
    parser.imp = {"test": set()}
    assert not parser.is_public("test.module")


# LLM-generated content at query #23
#--------------------------

```python
def test_isinstance_node_Try():
    node = Try(body=[], handlers=[], orelse=[], finalbody=[])
    assert isinstance(node, Try)


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_true():
    node = Call(func=Name(id='bool'), args=[])
    assert func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())


# LLM-generated content at query #25
#--------------------------

```python
def test_parse_basic_module():
    p = Parser()
    p.parse('test_module', 'def foo():\n    pass')
    assert 'test_module' in p.doc
    assert '# Module `test_module`' in p.doc['test_module']
    assert 'test_module' in p.level
    assert p.level['test_module'] == 0
    assert 'test_module' in p.root
    assert p.root['test_module'] == 'test_module'
    assert 'test_module' in p.imp
    assert len(p.imp['test_module']) == 0
    assert 'test_module.foo' in p.doc
    assert '# test_module.foo()' in p.doc['test_module.foo']
    assert 'test_module.foo' in p.level
    assert p.level['test_module.foo'] == 0
    assert 'test_module.foo' in p.root
    assert p.root['test_module.foo'] == 'test_module'

def test_parse_with_imports():
    p = Parser()
    p.parse('test_module', 'import os\nfrom sys import path\nx = 1')
    assert 'test_module' in p.alias
    assert p.alias['test_module.os'] == 'os'
    assert p.alias['test_module.path'] == 'sys.path'
    assert 'test_module.x' in p.alias
    assert p.alias['test_module.x'] == '1'
    assert 'test_module.x' in p.const
    assert p.const['test_module.x'] == 'int'

def test_parse_with_docstring():
    p = Parser()
    p.parse('test_module', '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass')
    assert 'test_module' in p.docstring
    assert 'Module docstring.' in p.docstring['test_module']
    assert 'test_module.foo' in p.docstring
    assert 'Function docstring.' in p.docstring['test_module.foo']

def test_parse_with_link_enabled():
    p = Parser(link=True)
    p.parse('test_module', 'def foo():\n    pass')
    assert '<a id="test_module"></a>' in p.doc['test_module']
    assert '<a id="test_module-foo"></a>' in p.doc['test_module.foo']

def test_parse_with_toc_enabled():
    p = Parser(toc=True)
    assert p.link is True
    p.parse('test_module', 'def foo():\n    pass')
    assert 'test_module' in p.doc
    assert 'test_module.foo' in p.doc

def test_parse_with_class_and_methods():
    p = Parser()
    p.parse('test_module', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass')
    assert 'test_module.Foo' in p.doc
    assert 'class test_module.Foo' in p.doc['test_module.Foo']
    assert 'test_module.Foo.bar' in p.doc
    assert 'test_module.Foo.baz' in p.doc
    assert 'test_module.Foo' in p.root
    assert p.root['test_module.Foo'] == 'test_module'
    assert 'test_module.Foo.bar' in p.root
    assert p.root['test_module.Foo.bar'] == 'test_module'
    assert 'test_module.Foo.baz' in p.root
    assert p.root['test_module.Foo.baz'] == 'test_module'


# LLM-generated content at query #26
#--------------------------

```python
def test__attr_returns_none_for_nonexistent_nested_attribute():
    class MockObject:
        pass

    obj = MockObject()
    assert _attr(obj, "nonexistent.attr") is None


# LLM-generated content at query #27
#--------------------------

```python
def test_globals_with_ann_assign():
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="TEST_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.root["test_module.TEST_CONST"] == "test_module"

def test_globals_with_assign():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="TEST_CONST", ctx=Store())],
        value=Constant(value=42),
        type_comment="int"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.root["test_module.TEST_CONST"] == "test_module"

def test_globals_with_assign_no_type_comment():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="TEST_CONST", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.root["test_module.TEST_CONST"] == "test_module"

def test_globals_with_all():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="public_func")], ctx=Load())
    )
    parser.globals(root, node)
    assert parser.imp["test_module"] == {"test_module.public_func"}

def test_globals_with_non_uppercase():
    parser = Parser()
    root = "test_module"
    node = Assign(
        targets=[Name(id="non_upper", ctx=Store())],
        value=Constant(value=42)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.non_upper"] == "42"
    assert "test_module.non_upper" not in parser.const
    assert "test_module.non_upper" not in parser.root


# LLM-generated content at query #28
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass", mode="eval").body]
    body = [
        parse("x: int", mode="eval").body,
        parse("y = 1", mode="eval").body,
        parse("del z", mode="eval").body
    ]
    parser.doc[name] = "# TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    parser.class_api(root, name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "z" not in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum", mode="eval").body]
    body = [
        parse("A = 1", mode="eval").body,
        parse("B = 2", mode="eval").body
    ]
    parser.doc[name] = "# TestEnum\n\n*Full name:* `test_module.TestEnum`\n\n"
    parser.class_api(root, name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "A" in parser.doc[name]
    assert "B" in parser.doc[name]
    assert "Members" not in parser.doc[name]


# LLM-generated content at query #29
#--------------------------

```python
def test_class_api_deletes_enum_attribute():
    parser = Parser()
    parser.doc = {}
    parser.alias = {}
    parser.const = {}
    parser.root = {}
    parser.level = {}
    parser.imp = {}
    parser.docstring = {}
    parser.b_level = 1
    parser.link = True
    parser.toc = False

    class_node = ClassDef(
        name='TestClass',
        bases=[Name(id='enum', ctx=Load())],
        body=[
            AnnAssign(
                target=Name(id='VALUE1', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=Constant(value=1)
            ),
            AnnAssign(
                target=Name(id='VALUE2', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=Constant(value=2)
            ),
            Delete(
                targets=[Name(id='VALUE1', ctx=Del())]
            )
        ],
        decorator_list=[]
    )

    parser.class_api('test_module', 'test_module.TestClass', class_node.bases, class_node.body)

    assert 'VALUE1' not in parser.doc['test_module.TestClass']
    assert 'VALUE2' in parser.doc['test_module.TestClass']


# LLM-generated content at query #30
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

def test_parser_post_init_with_toc():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #31
#--------------------------

```python
def test_is_public_with_all_l():
    p = Parser()
    p.imp = {'root': {'child', 'parent.child'}}
    p.root = {'parent.child': 'root'}
    assert p.is_public('parent.child')


# LLM-generated content at query #32
#--------------------------

```python
def test_imports_with_level():
    p = Parser()
    node = ImportFrom(module='os', names=[alias(name='path', asname=None)], level=1)
    p.imports('pkg.subpkg', node)
    assert p.alias['pkg.subpkg.path'] == 'os.path'


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_compile_empty():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.compile() == "\n"

def test_compile_with_toc():
    p = Parser.new(link=True, level=1, toc=True)
    p.doc["root"] = "# Module `{}`"
    p.docstring["root"] = "Root docstring"
    p.imp["root"] = set()
    p.root["root"] = "root"
    p.level["root"] = 0
    result = p.compile()
    assert "**Table of contents:**" in result
    assert "+ [root](#root)" in result
    assert "Root docstring" in result

def test_compile_without_toc():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root"] = "# Module `{}`"
    p.docstring["root"] = "Root docstring"
    p.imp["root"] = set()
    p.root["root"] = "root"
    p.level["root"] = 0
    result = p.compile()
    assert "**Table of contents:**" not in result
    assert "Root docstring" in result

def test_compile_with_magic_method():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root.__init__"] = "## __init__()\n\n*Full name:* `{}`"
    p.docstring["root.__init__"] = "Init docstring"
    p.imp["root"] = set()
    p.root["root.__init__"] = "root"
    p.level["root.__init__"] = 1
    result = p.compile()
    assert "Init docstring" in result

def test_compile_with_non_public():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root._private"] = "## _private()\n\n*Full name:* `{}`"
    p.docstring["root._private"] = "Private docstring"
    p.imp["root"] = set()
    p.root["root._private"] = "root"
    p.level["root._private"] = 1
    result = p.compile()
    assert "Private docstring" not in result

def test_compile_with_constants():
    p = Parser.new(link=False, level=1, toc=False)
    p.doc["root"] = "# Module `{}`"
    p.const["root.CONST"] = "int"
    p.imp["root"] = set()
    p.root["root"] = "root"
    p.root["root.CONST"] = "root"
    p.level["root"] = 0
    result = p.compile()
    assert "Constants" in result
    assert "CONST" in result
    assert "int" in result


# LLM-generated content at query #35
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

def test_parser_new_classmethod():
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


# LLM-generated content at query #36
#--------------------------

```python
def test_func_api_with_no_args_and_no_return():
    parser = Parser()
    node = arguments([], None, [], [], None, [], None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |'

def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments([arg('x', None), arg('y', None)], None, [], [], None, [], None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|x|y|return|\n|:---:|:---:|:---:|\n| | | |'

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments([arg('x', None), arg('y', None)], None, [arg('z', None)], [Constant(1)], None, [], None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|x|y|z|return|\n|:---:|:---:|:---:|:---:|\n| | |`1`| |\n| | | | |'

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments([], None, [], [], arg('args', None), [], None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|*args|return|\n|:---:|:---:|\n| | |'

def test_func_api_with_kwonlyargs():
    parser = Parser()
    node = arguments([], None, [], [], None, [arg('x', None), arg('y', None)], None)
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|*|x|y|return|\n|:---:|:---:|:---:|:---:|\n| | | | |'

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments([], None, [], [], None, [], arg('kwargs', None))
    parser.func_api('root', 'test_func', node, None, has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|**kwargs|return|\n|:---:|:---:|\n| | |'

def test_func_api_with_return_annotation():
    parser = Parser()
    node = arguments([], None, [], [], None, [], None)
    parser.func_api('root', 'test_func', node, Constant('int'), has_self=False, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n|`int`|'

def test_func_api_with_has_self():
    parser = Parser()
    node = arguments([arg('self', None)], None, [], [], None, [], None)
    parser.func_api('root', 'test_func', node, None, has_self=True, cls_method=False)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |'

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments([arg('cls', None)], None, [], [], None, [], None)
    parser.func_api('root', 'test_func', node, None, has_self=True, cls_method=True)
    assert parser.doc['root.test_func'] == '#' * (parser.b_level + 2) + ' test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n|return|\n|:---:|\n| |'


# LLM-generated content at query #37
#--------------------------

```python
def test_imports_without_asname():
    p = Parser()
    node = ImportFrom(module="sys", names=[alias(name="path")], level=0)
    p.imports("pkg", node)
    assert p.alias["pkg.path"] == "sys.path"


# LLM-generated content at query #38
#--------------------------

```python
def test_load_docstring_basic():
    p = Parser()
    p.doc = {'pkg.submod.func': 'Function doc', 'pkg.submod': 'Module doc'}
    p.docstring = {}
    p.root = {'pkg.submod.func': 'pkg.submod', 'pkg.submod': 'pkg.submod'}
    m = ModuleType('pkg.submod')
    m.func = lambda: None
    m.func.__doc__ = 'New function doc'
    p.load_docstring('pkg.submod', m)
    assert p.docstring['pkg.submod.func'] == 'New function doc'
    assert 'pkg.submod' not in p.docstring

def test_load_docstring_nested():
    p = Parser()
    p.doc = {'pkg.submod.Class.method': 'Method doc', 'pkg.submod.Class': 'Class doc'}
    p.docstring = {}
    p.root = {'pkg.submod.Class.method': 'pkg.submod', 'pkg.submod.Class': 'pkg.submod'}
    m = ModuleType('pkg.submod')
    m.Class = type('Class', (), {'method': lambda: None})
    m.Class.method.__doc__ = 'New method doc'
    p.load_docstring('pkg.submod', m)
    assert p.docstring['pkg.submod.Class.method'] == 'New method doc'
    assert 'pkg.submod.Class' not in p.docstring

def test_load_docstring_none_doc():
    p = Parser()
    p.doc = {'pkg.submod.func': 'Function doc'}
    p.docstring = {}
    p.root = {'pkg.submod.func': 'pkg.submod'}
    m = ModuleType('pkg.submod')
    m.func = lambda: None
    p.load_docstring('pkg.submod', m)
    assert 'pkg.submod.func' not in p.docstring

def test_load_docstring_no_match():
    p = Parser()
    p.doc = {'pkg.submod.func': 'Function doc'}
    p.docstring = {}
    p.root = {'pkg.submod.func': 'pkg.submod'}
    m = ModuleType('pkg.submod')
    m.other_func = lambda: None
    m.other_func.__doc__ = 'Other doc'
    p.load_docstring('pkg.submod', m)
    assert 'pkg.submod.func' not in p.docstring
    assert 'pkg.submod.other_func' not in p.docstring


# LLM-generated content at query #39
#--------------------------

```python
def test_func_ann_with_self_and_annotation():
    parser = Parser()
    args = [arg('self', Name(id='SomeClass', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=False))
    assert result == ['Self', 'int']

def test_func_ann_with_cls_method():
    parser = Parser()
    args = [arg('cls', Name(id='SomeClass', ctx=Load())), arg('x', Name(id='int', ctx=Load()))]
    result = list(parser.func_ann('module', args, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int']

def test_func_ann_without_annotation():
    parser = Parser()
    args = [arg('x', None), arg('y', None)]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['ANY', 'ANY']

def test_func_ann_with_star_arg():
    parser = Parser()
    args = [arg('x', Name(id='int', ctx=Load())), arg('*', None), arg('y', Name(id='str', ctx=Load()))]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', '', 'str']


# LLM-generated content at query #40
#--------------------------

```python
def test__attr_with_single_attribute():
    class TestObj:
        attr1 = "value1"
    obj = TestObj()
    assert _attr(obj, "attr1") == "value1"

def test__attr_with_nested_attributes():
    class TestObj:
        class Nested:
            attr2 = "value2"
        attr1 = Nested()
    obj = TestObj()
    assert _attr(obj, "attr1.attr2") == "value2"

def test__attr_with_nonexistent_attribute():
    class TestObj:
        attr1 = "value1"
    obj = TestObj()
    assert _attr(obj, "nonexistent") is None

def test__attr_with_partial_nonexistent_nested_attribute():
    class TestObj:
        class Nested:
            attr2 = "value2"
        attr1 = Nested()
    obj = TestObj()
    assert _attr(obj, "attr1.nonexistent") is None

def test__attr_with_deeply_nested_attributes():
    class TestObj:
        class Level1:
            class Level2:
                attr3 = "value3"
            attr2 = Level2()
        attr1 = Level1()
    obj = TestObj()
    assert _attr(obj, "attr1.attr2.attr3") == "value3"

def test__attr_with_empty_attribute_string():
    class TestObj:
        attr1 = "value1"
    obj = TestObj()
    assert _attr(obj, "") is None


# LLM-generated content at query #41
#--------------------------

```python
def test_func_api_with_kwonlyargs():
    parser = Parser()
    root = "test_module"
    name = "test_function"
    node = arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[arg("kw_arg", None)],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    assert arg('*', None) in parser.doc[name]


# LLM-generated content at query #42
#--------------------------

```python
def test_class_api_with_enum_base():
    parser = Parser()
    parser.root = {'test.EnumClass': 'test'}
    parser.level = {'test.EnumClass': 1}
    parser.doc = {'test.EnumClass': '# test.EnumClass\n\n'}
    parser.alias = {}
    parser.const = {}
    parser.imp = {'test': set()}

    class_def = ClassDef(
        name='EnumClass',
        bases=[Name(id='Enum', ctx=Load())],
        body=[
            AnnAssign(
                target=Name(id='RED', ctx=Store()),
                annotation=Name(id='int', ctx=Load()),
                value=Constant(value=1)
            )
        ]
    )

    parser.class_api('test', 'test.EnumClass', class_def.bases, class_def.body)
    assert 'Enums' in parser.doc['test.EnumClass']
    assert 'RED' in parser.doc['test.EnumClass']


# LLM-generated content at query #43
#--------------------------

```python
def test_api_no_recursion_on_non_class():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
    parser.api(root, node)
    assert len(parser.doc) == 1
    assert "test_module.test_func" in parser.doc


# LLM-generated content at query #44
#--------------------------

```python
def test_func_ann_predicate():
    parser = Parser()
    args = [arg('self', None)]
    result = list(parser.func_ann('root', args, has_self=True, cls_method=False))
    assert result[0] == 'Self'


# LLM-generated content at query #45
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
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True

def test_parser_post_init_toc_false():
    p = Parser.new(link=False, level=1, toc=False)
    assert p.link is False
    assert p.b_level == 1
    assert p.toc is False


# LLM-generated content at query #46
#--------------------------

```python
def test_globals_const_not_any():
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="CONST", ctx=Store()),
        annotation=Constant(value="int"),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.const.get("test_module.CONST", "ANY") != "ANY"


# LLM-generated content at query #47
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser()
    node = arguments(posonlyargs=[arg('a', None)], args=[arg('b', None)], defaults=[])
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| a | b | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `Any` | `Any` | `Any` |\n\n'
    )

def test_func_api_with_keyword_args():
    parser = Parser()
    node = arguments(args=[], kwonlyargs=[arg('a', None), arg('b', None)], kw_defaults=[])
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| * | a | b | return |\n'
        '|:---:|:---:|:---:|:---:|\n'
        '|  | `Any` | `Any` | `Any` |\n\n'
    )

def test_func_api_with_varargs():
    parser = Parser()
    node = arguments(args=[arg('a', None)], vararg=arg('args', None), defaults=[])
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| a | *args | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `Any` | `Any` | `Any` |\n\n'
    )

def test_func_api_with_kwargs():
    parser = Parser()
    node = arguments(args=[], kwarg=arg('kwargs', None))
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| **kwargs | return |\n'
        '|:---:|:---:|\n'
        '| `Any` | `Any` |\n\n'
    )

def test_func_api_with_defaults():
    parser = Parser()
    node = arguments(args=[arg('a', None), arg('b', None)], defaults=[Constant(1)])
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| a | b | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `Any` | `1` | `Any` |\n\n'
    )

def test_func_api_with_self():
    parser = Parser()
    node = arguments(args=[arg('self', None)], defaults=[])
    parser.func_api('root', 'name', node, None, has_self=True, cls_method=False)
    assert parser.doc['name'] == (
        '| self | return |\n'
        '|:---:|:---:|\n'
        '| `Self` | `Any` |\n\n'
    )

def test_func_api_with_cls_method():
    parser = Parser()
    node = arguments(args=[arg('cls', None)], defaults=[])
    parser.func_api('root', 'name', node, None, has_self=True, cls_method=True)
    assert parser.doc['name'] == (
        '| cls | return |\n'
        '|:---:|:---:|\n'
        '| `type[Self]` | `Any` |\n\n'
    )

def test_func_api_with_annotations():
    parser = Parser()
    node = arguments(args=[arg('a', Name('int', None)), arg('b', Name('str', None))], defaults=[])
    parser.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| a | b | return |\n'
        '|:---:|:---:|:---:|\n'
        '| `int` | `str` | `Any` |\n\n'
    )

def test_func_api_with_return_annotation():
    parser = Parser()
    node = arguments(args=[], defaults=[])
    parser.func_api('root', 'name', node, Name('bool', None), has_self=False, cls_method=False)
    assert parser.doc['name'] == (
        '| return |\n'
        '|:---:|\n'
        '| `bool` |\n\n'
    )


# LLM-generated content at query #48
#--------------------------

```python
def test_func_api_with_kwarg():
    parser = Parser()
    root = "test_module"
    name = "test_function"
    node = arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=arg('kwargs', None), defaults=[])
    returns = None
    has_self = False
    cls_method = False
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert '**kwargs' in parser.doc[name]


# LLM-generated content at query #49
#--------------------------

```python
def test__e_type_empty_input():
    assert _e_type() == ""

def test__e_type_with_none_element():
    assert _e_type([None]) == ""
    assert _e_type([None, None]) == ""

def test__e_type_with_non_constant_element():
    assert _e_type([1]) == ""
    assert _e_type([1, 2]) == ""

def test__e_type_with_constant_elements():
    assert _e_type([Constant(1)]) == "[int]"
    assert _e_type([Constant(1), Constant(2)]) == "[int]"

def test__e_type_with_mixed_constant_types():
    assert _e_type([Constant(1), Constant("a")]) == "[Any]"
    assert _e_type([Constant(1.0), Constant(2)]) == "[Any]"

def test__e_type_with_multiple_sequences():
    assert _e_type([Constant(1)], [Constant(2)]) == "[int, int]"
    assert _e_type([Constant(1)], [Constant("a")]) == "[int, str]"
    assert _e_type([Constant(1)], [Constant(1.0)]) == "[int, Any]"


# LLM-generated content at query #50
#--------------------------

```python
def test_none_attribute_access():
    class Dummy:
        pass
    obj = Dummy()
    assert _attr(obj, 'nonexistent') is None


# LLM-generated content at query #51
#--------------------------

```python
def test_imports_with_asname():
    p = Parser()
    node = Import(names=[alias(name='module', asname='alias')])
    p.imports('root', node)
    assert p.alias.get('root.alias') == 'module'


# LLM-generated content at query #52
#--------------------------

```python
def test_walk_body_with_try_node():
    try_node = Try(
        body=[stmt()],
        handlers=[ExceptHandler(body=[stmt()])],
        orelse=[stmt()],
        finalbody=[stmt()]
    )
    result = list(walk_body([try_node]))
    assert len(result) == 4


# LLM-generated content at query #53
#--------------------------

```python
def test_imports_with_asname():
    p = Parser()
    node = Import(names=[alias(name='os', asname='operating_system')])
    p.imports('pkg', node)
    assert p.alias.get('pkg.operating_system') == 'os'


# LLM-generated content at query #54
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


# LLM-generated content at query #55
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

def test_parser_post_init_toc_true():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #56
#--------------------------

```python
def test_visit_Name_with_self_ty():
    resolver = Resolver(root="test", alias={}, self_ty="MyClass")
    node = Name("MyClass", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

def test_visit_Name_with_alias():
    resolver = Resolver(root="test", alias={"test.MyType": "int"})
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "int"

def test_visit_Name_without_alias():
    resolver = Resolver(root="test", alias={})
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"

def test_visit_Name_with_TypeVar_alias():
    resolver = Resolver(root="test", alias={"test.MyType": "typing.TypeVar('T')"})
    node = Name("MyType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "MyType"


# LLM-generated content at query #57
#--------------------------

```python
def test_is_public_with_all_l():
    parser = Parser()
    parser.imp = {'pkg': {'subpkg', 'subpkg.submod'}}
    parser.root = {'pkg.subpkg': 'pkg', 'pkg.subpkg.submod': 'pkg'}
    assert parser.is_public('pkg.subpkg') is True


# LLM-generated content at query #58
#--------------------------

```python
def test_class_api():
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


# LLM-generated content at query #59
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [parse("BaseClass", mode="eval").body]
    body = [
        parse("x: int = 1", mode="eval").body,
        parse("y: str = 'hello'", mode="eval").body,
        parse("z = [1, 2, 3]", mode="eval").body,
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = "# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id=\"test-module-testclass\"></a>\n\n"
    expected_doc += table("Bases", items=["BaseClass"])
    expected_doc += table("Members", "Type", items=[("x", "int"), ("y", "str"), ("z", "list[int]")])
    assert parser.doc[name] == expected_doc

def test_class_api_with_enum():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [parse("enum.Enum", mode="eval").body]
    body = [
        parse("A = 1", mode="eval").body,
        parse("B = 2", mode="eval").body,
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = "# class TestEnum\n\n*Full name:* `test_module.TestEnum`\n<a id=\"test-module-testenum\"></a>\n\n"
    expected_doc += table("Enums", items=["A", "B"])
    assert parser.doc[name] == expected_doc

def test_class_api_with_deleted_member():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = []
    body = [
        parse("x: int = 1", mode="eval").body,
        parse("del x", mode="eval").body,
    ]
    parser.class_api(root, name, bases, body)
    expected_doc = "# class TestClass\n\n*Full name:* `test_module.TestClass`\n<a id=\"test-module-testclass\"></a>\n\n"
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #60
#--------------------------

```python
def test_load_docstring():
    p = Parser()
    p.doc = {'pkg': '# Module `pkg`', 'pkg.func': '## func()'}
    p.docstring = {}
    m = ModuleType('pkg')
    m.__doc__ = "Package doc"
    m.func = lambda: None
    m.func.__doc__ = "Function doc"
    p.load_docstring('pkg', m)
    assert p.docstring['pkg'] == "Package doc"
    assert p.docstring['pkg.func'] == "Function doc"


# LLM-generated content at query #61
#--------------------------

```python
def test_is_magic_predicate():
    parser = Parser()
    parser.doc = {"__init__": "# Module `__init__`\n\n"}
    parser.docstring = {}
    parser.imp = {}
    parser.root = {"__init__": "__init__"}
    parser.level = {"__init__": 0}
    parser.toc = False
    result = parser.compile()
    assert "__init__" not in result


# LLM-generated content at query #62
#--------------------------

```python
def test_globals_predicate_false():
    parser = Parser()
    node = Assign(targets=[Name(id='__all__')], value=Tuple(elts=[Constant(value='foo')]))
    parser.globals('root', node)
    assert '__all__' not in parser.imp['root']


# LLM-generated content at query #63
#--------------------------

```python
def test_func_api_with_positional_args():
    parser = Parser.new(link=False, level=1, toc=False)
    node = arguments(
        posonlyargs=[arg('a', None), arg('b', None)],
        args=[arg('c', None)],
        defaults=[Constant(1), Constant(2)],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', node, None, has_self=False, cls_method=False)
    assert '| a | b | / | c | return |\n|:---:|:---:|:---:|:---:|:---:|\n| `int` | `int` |  |  |  |\n| 1 | 2 |  |  |  |' in parser.doc['root.func']

def test_func_api_with_keyword_args():
    parser = Parser.new(link=False, level=1, toc=False)
    node = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[arg('a', None), arg('b', None)],
        kw_defaults=[Constant(1), Constant(2)],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', node, None, has_self=False, cls_method=False)
    assert '| * | a | b | return |\n|:---:|:---:|:---:|:---:|\n|  | `int` | `int` |  |\n|  | 1 | 2 |  |' in parser.doc['root.func']

def test_func_api_with_varargs():
    parser = Parser.new(link=False, level=1, toc=False)
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=arg('kwargs', None),
        vararg=arg('args', None)
    )
    parser.func_api('root', 'root.func', node, None, has_self=False, cls_method=False)
    assert '| a | *args | **kwargs | return |\n|:---:|:---:|:---:|:---:|\n|  |  |  |  |' in parser.doc['root.func']

def test_func_api_with_self_and_cls_method():
    parser = Parser.new(link=False, level=1, toc=False)
    node = arguments(
        posonlyargs=[],
        args=[arg('self', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', node, None, has_self=True, cls_method=True)
    assert '| type[Self] | a | return |\n|:---:|:---:|:---:|\n|  |  |  |' in parser.doc['root.func']

def test_func_api_with_return_annotation():
    parser = Parser.new(link=False, level=1, toc=False)
    node = arguments(
        posonlyargs=[],
        args=[arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'root.func', node, Constant('str'), has_self=False, cls_method=False)
    assert '| a | return |\n|:---:|:---:|\n|  | `str` |' in parser.doc['root.func']


# LLM-generated content at query #64
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

def test_parser_post_init_with_toc():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #65
#--------------------------

```python
def test_func_api_with_vararg():
    parser = Parser()
    root = "module"
    name = "module.func"
    node = arguments(
        posonlyargs=[],
        args=[arg("x")],
        vararg=arg("args"),
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False
    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)
    assert parser.doc[name].endswith("| Arguments | Type |\n| --- | --- |\n| x | Any |\n| *args | Any |\n| return | Any |\n")


# LLM-generated content at query #66
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


# LLM-generated content at query #67
#--------------------------

```python
def test_load_docstring_updates_docstring():
    p = Parser()
    p.doc = {'pkg.sub': '...', 'pkg.other': '...'}
    p.docstring = {}
    m = type('Module', (), {'sub': 'Sub module doc'})()
    p.load_docstring('pkg', m)
    assert 'pkg.sub' in p.docstring
    assert p.docstring['pkg.sub'] == 'Sub module doc'


# LLM-generated content at query #68
#--------------------------

```python
def test__attr_returns_none_for_missing_nested_attribute():
    class MockObject:
        pass

    obj = MockObject()
    result = _attr(obj, "non.existent.attribute")
    assert result is None


# LLM-generated content at query #69
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

def test_parser_post_init_with_toc():
    parser = Parser.new(link=False, level=1, toc=True)
    assert parser.link is True
    assert parser.b_level == 1
    assert parser.toc is True


# LLM-generated content at query #70
#--------------------------

```python
def test_class_api_with_bases_and_members():
    parser = Parser()
    name = "test.module.ClassName"
    bases = [Name(id="BaseClass", ctx=Load())]
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name(id="another_attr", ctx=Store())],
            value=Constant(value=42)
        )
    ]
    parser.doc[name] = "# ClassName\n\n*Full name:* `test.module.ClassName`\n\n"
    parser.class_api("test.module", name, bases, body)
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "public_attr" in parser.doc[name]
    assert "another_attr" in parser.doc[name]

def test_class_api_with_enum():
    parser = Parser()
    name = "test.module.EnumClass"
    bases = [Name(id="enum.Enum", ctx=Load())]
    body = [
        AnnAssign(
            target=Name(id="FIRST", ctx=Store()),
            annotation=Subscript(
                value=Name(id="int", ctx=Load()),
                slice=Constant(value=1),
                ctx=Load()
            ),
            value=None,
            simple=1
        ),
        AnnAssign(
            target=Name(id="SECOND", ctx=Store()),
            annotation=Subscript(
                value=Name(id="int", ctx=Load()),
                slice=Constant(value=2),
                ctx=Load()
            ),
            value=None,
            simple=1
        )
    ]
    parser.doc[name] = "# EnumClass\n\n*Full name:* `test.module.EnumClass`\n\n"
    parser.class_api("test.module", name, bases, body)
    assert "Enums" in parser.doc[name]
    assert "FIRST" in parser.doc[name]
    assert "SECOND" in parser.doc[name]
    assert "Members" not in parser.doc[name]

def test_class_api_with_deleted_members():
    parser = Parser()
    name = "test.module.ClassName"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="public_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        Delete(targets=[Name(id="public_attr", ctx=Del())])
    ]
    parser.doc[name] = "# ClassName\n\n*Full name:* `test.module.ClassName`\n\n"
    parser.class_api("test.module", name, bases, body)
    assert "Members" not in parser.doc[name]
    assert "public_attr" not in parser.doc[name]

def test_class_api_with_private_members():
    parser = Parser()
    name = "test.module.ClassName"
    bases = []
    body = [
        AnnAssign(
            target=Name(id="_private_attr", ctx=Store()),
            annotation=Name(id="int", ctx=Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name(id="public_attr", ctx=Store())],
            value=Constant(value=42)
        )
    ]
    parser.doc[name] = "# ClassName\n\n*Full name:* `test.module.ClassName`\n\n"
    parser.class_api("test.module", name, bases, body)
    assert "_private_attr" not in parser.doc[name]
    assert "public_attr" in parser.doc[name]


# LLM-generated content at query #71
#--------------------------

```python
def test_func_api_with_kwonlyargs():
    p = Parser()
    node = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        vararg=None,
        kwonlyargs=[arg('a'), arg('b')],
        kw_defaults=[Constant(1), Constant(2)],
        kwarg=None
    )
    p.func_api('root', 'name', node, None, has_self=False, cls_method=False)
    assert '*' in p.doc['root.name']


# LLM-generated content at query #72
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

def test_parser_post_init_with_toc():
    p = Parser.new(link=False, level=1, toc=True)
    assert p.link is True
    assert p.b_level == 1
    assert p.toc is True


# LLM-generated content at query #73
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
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert '| a | b | / | c | return |' in parser.doc['root.func']
    assert '|:---:|:---:|:---:|:---:|:---:|' in parser.doc['root.func']
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
        kwarg=arg('kwargs', None),
        vararg=arg('args', None)
    )
    parser.func_api('root', 'func', node, None, has_self=False, cls_method=False)
    assert '| a | * | b | **kwargs | return |' in parser.doc['root.func']
    assert '|:---:|:---:|:---:|:---:|:---:|' in parser.doc['root.func']
    assert '| `a` |  | `b` |  | `Any` |' in parser.doc['root.func']

def test_func_api_with_self_and_cls_method():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[arg('self', None), arg('a', None)],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    parser.func_api('root', 'func', node, None, has_self=True, cls_method=True)
    assert '| self | a | return |' in parser.doc['root.func']
    assert '|:---:|:---:|:---:|' in parser.doc['root.func']
    assert '| `type[Self]` | `Any` | `Any` |' in parser.doc['root.func']

def test_func_api_with_returns():
    parser = Parser()
    node = arguments(
        posonlyargs=[],
        args=[],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        vararg=None
    )
    returns = Name(id='int', ctx=Load())
    parser.func_api('root', 'func', node, returns, has_self=False, cls_method=False)
    assert '| return |' in parser.doc['root.func']
    assert '|:---:|' in parser.doc['root.func']
    assert '| `int` |' in parser.doc['root.func']


# LLM-generated content at query #74
#--------------------------

```python
def test_predicate_at_line_11_evaluates_to_true():
    node = Call(func=Name(id='bool'))
    assert func in chain({'bool', 'int', 'float', 'complex', 'str'}, PEP585.keys(), PEP585.values())


