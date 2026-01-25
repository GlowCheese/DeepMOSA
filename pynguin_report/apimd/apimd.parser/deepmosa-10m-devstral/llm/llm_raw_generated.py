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


