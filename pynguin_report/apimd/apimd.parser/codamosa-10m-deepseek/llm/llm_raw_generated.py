####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.ExampleClass"
    bases = [Constant("object")]
    body = [
        AnnAssign(Name("attr1"), Constant("int"), Constant(42), simple=True),
        Assign([Name("attr2")], Constant("str"), type_comment="str"),
        Delete([Name("attr1")]),
        AnnAssign(Name("attr3"), Constant("float"), Constant(3.14), simple=True),
    ]
    
    parser.class_api(root, name, bases, body)
    
    assert parser.doc[name].startswith("### class ExampleClass\n\n")
    assert "Bases" in parser.doc[name]
    assert "object" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr3" in parser.doc[name]
    assert "float" in parser.doc[name]
    assert "attr1" not in parser.doc[name]
    assert "attr2" not in parser.doc[name]
    
    # Test with enum bases
    enum_bases = [Constant("enum.Enum")]
    enum_body = [
        AnnAssign(Name("ENUM_VALUE1"), Constant("int"), Constant(1), simple=True),
        AnnAssign(Name("ENUM_VALUE2"), Constant("int"), Constant(2), simple=True),
    ]
    
    parser.class_api(root, "test_module.ExampleEnum", enum_bases, enum_body)
    
    assert "Enums" in parser.doc["test_module.ExampleEnum"]
    assert "ENUM_VALUE1" in parser.doc["test_module.ExampleEnum"]
    assert "ENUM_VALUE2" in parser.doc["test_module.ExampleEnum"]


# LLM-generated content at query #2
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver(root="example", alias={}, self_ty="Self")
    
    # Test with non-string constant
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string constant that is a valid name
    node = Constant(value="example_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "example_name"
    
    # Test with string constant that is not a valid name
    node = Constant(value="invalid name with spaces")
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string constant that is a valid expression
    node = Constant(value="example_name + 1")
    result = resolver.visit_Constant(node)
    assert isinstance(result, BinOp)


# LLM-generated content at query #3
#--------------------------

```python
def test_doctest():
    # Test case 1: Simple doctest with one line
    doc = ">>> print('Hello, World!')"
    expected = "```python\n>>> print('Hello, World!')\n```"
    assert doctest(doc) == expected

    # Test case 2: Doctest with multiple lines
    doc = ">>> x = 5\n>>> y = 10\n>>> print(x + y)"
    expected = "```python\n>>> x = 5\n>>> y = 10\n>>> print(x + y)\n```"
    assert doctest(doc) == expected

    # Test case 3: Doctest with non-doctest lines
    doc = "This is not a doctest.\n>>> print('This is a doctest.')"
    expected = "This is not a doctest.\n```python\n>>> print('This is a doctest.')\n```"
    assert doctest(doc) == expected

    # Test case 4: Doctest with closing code block at the end
    doc = ">>> print('End of doctest')"
    expected = "```python\n>>> print('End of doctest')\n```"
    assert doctest(doc) == expected

    # Test case 5: Empty string
    doc = ""
    expected = ""
    assert doctest(doc) == expected

    # Test case 6: Doctest with only non-doctest lines
    doc = "This is just text.\nMore text."
    expected = "This is just text.\nMore text."
    assert doctest(doc) == expected

    # Test case 7: Doctest with mixed lines and multiple code blocks
    doc = "Text before doctest.\n>>> print('Doctest line 1')\nText between.\n>>> print('Doctest line 2')"
    expected = "Text before doctest.\n```python\n>>> print('Doctest line 1')\n```\nText between.\n```python\n>>> print('Doctest line 2')\n```"
    assert doctest(doc) == expected


# LLM-generated content at query #4
#--------------------------

Here's a unit test for the `walk_body` function using pytest:


# LLM-generated content at query #5
#--------------------------

def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with Name target and value
    ann_assign_node = AnnAssign(
        target=Name(id="TEST_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign_node)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"

    # Test Assign with Name target and value
    assign_node = Assign(
        targets=[Name(id="ANOTHER_CONST", ctx=Store())],
        value=Constant(value="hello")
    )
    parser.globals(root, assign_node)
    assert parser.alias["test_module.ANOTHER_CONST"] == "'hello'"
    assert parser.const["test_module.ANOTHER_CONST"] == "str"

    # Test Assign with Tuple target and value
    tuple_assign_node = Assign(
        targets=[Tuple(elts=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())], ctx=Store())],
        value=Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    )
    parser.globals(root, tuple_assign_node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias

    # Test Assign with __all__ as target and List value
    all_assign_node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load())
    )
    parser.globals(root, all_assign_node)
    assert "test_module.func1" in parser.imp[root]
    assert "test_module.func2" in parser.imp[root]

    # Test AnnAssign with non-Name target
    non_name_ann_assign_node = AnnAssign(
        target=Tuple(elts=[Name(id="x", ctx=Store()), Name(id="y", ctx=Store())], ctx=Store()),
        annotation=Name(id="tuple", ctx=Load()),
        value=Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load()),
        simple=0
    )
    parser.globals(root, non_name_ann_assign_node)
    assert "test_module.x" not in parser.alias
    assert "test_module.y" not in parser.alias

    # Test Assign with multiple targets
    multi_target_assign_node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=3)
    )
    parser.globals(root, multi_target_assign_node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #6
#--------------------------

```python
def test_Parser_parse():
    parser = Parser()
    root = "test_module"
    script = """
import os
from typing import List

CONSTANT = 42

def func(a: int, b: str) -> str:
    \"\"\"Example function.\"\"\"
    return b

class Example:
    \"\"\"Example class.\"\"\"
    def method(self, x: List[int]) -> None:
        pass
    """
    parser.parse(root, script)
    
    assert root in parser.doc
    assert root in parser.level
    assert root in parser.imp
    assert root in parser.root
    assert "test_module.CONSTANT" in parser.const
    assert "test_module.func" in parser.doc
    assert "test_module.Example" in parser.doc
    assert "test_module.Example.method" in parser.doc
    assert "test_module.func" in parser.docstring
    assert "test_module.Example" in parser.docstring
    assert "test_module.Example.method" in parser.docstring
    assert "os" in parser.alias
    assert "List" in parser.alias


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_globals():
    # Test with AnnAssign
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="TEST_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.alias["test_module.TEST_CONST"] == "42"

    # Test with Assign and type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="ANOTHER_CONST", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.const["test_module.ANOTHER_CONST"] == "str"
    assert parser.alias["test_module.ANOTHER_CONST"] == "'hello'"

    # Test with Assign without type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="NO_TYPE_CONST", ctx=Store())],
        value=Constant(value=True)
    )
    parser.globals(root, node)
    assert parser.const["test_module.NO_TYPE_CONST"] == "bool"
    assert parser.alias["test_module.NO_TYPE_CONST"] == "True"

    # Test with __all__ assignment
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="TEST_CONST"), Constant(value="func")])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.TEST_CONST", "test_module.func"}

    # Test with non-constant __all__ value (should be ignored)
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Name(id="some_list", ctx=Load())
    )
    parser.globals(root, node)
    assert not parser.imp[root]

    # Test with multiple targets (should be ignored)
    parser = Parser()
    node = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert not parser.const
    assert not parser.alias

    # Test with non-Name target (should be ignored)
    parser = Parser()
    node = Assign(
        targets=[Attribute(value=Name(id="mod", ctx=Load()), attr="name", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert not parser.const
    assert not parser.alias


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with Name target
    ann_assign = AnnAssign(
        target=Name(id="TEST_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert parser.alias["test_module.TEST_CONST"] == "42"
    assert parser.const["test_module.TEST_CONST"] == "int"
    
    # Test Assign with Name target and type comment
    assign = Assign(
        targets=[Name(id="TEST_CONST_2", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, assign)
    assert parser.alias["test_module.TEST_CONST_2"] == "'hello'"
    assert parser.const["test_module.TEST_CONST_2"] == "str"
    
    # Test Assign with Name target and no type comment
    assign_no_type_comment = Assign(
        targets=[Name(id="TEST_CONST_3", ctx=Store())],
        value=Constant(value=3.14)
    )
    parser.globals(root, assign_no_type_comment)
    assert parser.alias["test_module.TEST_CONST_3"] == "3.14"
    assert parser.const["test_module.TEST_CONST_3"] == "float"
    
    # Test Assign with non-Name target
    assign_non_name = Assign(
        targets=[Subscript(
            value=Name(id="list", ctx=Load()),
            slice=Constant(value=0),
            ctx=Store()
        )],
        value=Constant(value=1)
    )
    parser.globals(root, assign_non_name)
    assert "test_module.list" not in parser.alias
    
    # Test __all__ assignment
    __all__ = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="TEST_CONST")], ctx=Load())
    )
    parser.globals(root, __all__)
    assert "test_module.TEST_CONST" in parser.imp[root]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_class_api():
    # Setup
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Constant("object")]
    body = [
        AnnAssign(Name("attr1"), annotation=Name("int"), value=Constant(42)),
        AnnAssign(Name("attr2"), annotation=Name("str"), value=Constant("hello")),
        Assign([Name("attr3")], Constant(3.14)),
        Delete([Name("attr1")])
    ]

    # Execute
    parser.class_api(root, name, bases, body)

    # Assert
    assert parser.doc[name] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    assert parser.doc[name] + table("Members", "Type", items=[
        (code("attr2"), code("str")),
        (code("attr3"), code("float"))
    ]) in parser.doc[name]


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    
    # Test Import
    import_node = Import(names=[alias(name='os', asname=None), alias(name='sys', asname=None)])
    parser.imports('test_root', import_node)
    assert parser.alias['test_root.os'] == 'os'
    assert parser.alias['test_root.sys'] == 'sys'
    
    # Test Import with alias
    import_node = Import(names=[alias(name='os', asname='operating_system')])
    parser.imports('test_root', import_node)
    assert parser.alias['test_root.operating_system'] == 'os'
    
    # Test ImportFrom with relative level
    import_from_node = ImportFrom(module='submodule', names=[alias(name='func', asname=None)], level=1)
    parser.imports('test_root.module', import_from_node)
    assert parser.alias['test_root.module.func'] == 'test_root.submodule.func'
    
    # Test ImportFrom without relative level
    import_from_node = ImportFrom(module='submodule', names=[alias(name='func', asname='f')], level=0)
    parser.imports('test_root', import_from_node)
    assert parser.alias['test_root.f'] == 'submodule.func'
    
    # Test ImportFrom with asname
    import_from_node = ImportFrom(module='submodule', names=[alias(name='func', asname='f')], level=1)
    parser.imports('test_root.module', import_from_node)
    assert parser.alias['test_root.module.f'] == 'test_root.submodule.func'


# LLM-generated content at query #11
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver(root="test_module", alias={})
    
    # Test with non-string Constant
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string Constant that is not a valid name
    node = Constant(value="not_a_valid_name")
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string Constant that is a valid name
    node = Constant(value="valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"
    assert isinstance(result.ctx, Load)
    
    # Test with string Constant that raises SyntaxError when parsed
    node = Constant(value="invalid syntax")
    result = resolver.visit_Constant(node)
    assert result == node


# LLM-generated content at query #12
#--------------------------

```python
def test_Parser_globals():
    # Test with AnnAssign (type annotated assignment)
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

    # Test with Assign (regular assignment)
    parser = Parser()
    node = Assign(
        targets=[Name(id="ANOTHER_CONST", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.ANOTHER_CONST"] == "'hello'"
    assert parser.const["test_module.ANOTHER_CONST"] == "str"

    # Test with Assign without type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="NO_TYPE_CONST", ctx=Store())],
        value=Constant(value=True)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.NO_TYPE_CONST"] == "True"
    assert parser.const["test_module.NO_TYPE_CONST"] == "bool"

    # Test with __all__ assignment
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="PUBLIC_FUNC"),
            Constant(value="PublicClass")
        ], ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.PUBLIC_FUNC" in parser.imp["test_module"]
    assert "test_module.PublicClass" in parser.imp["test_module"]

    # Test with non-Name target (should be ignored)
    parser = Parser()
    initial_alias_count = len(parser.alias)
    node = Assign(
        targets=[Subscript(
            value=Name(id="some_list", ctx=Load()),
            slice=Constant(value=0),
            ctx=Store()
        )],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert len(parser.alias) == initial_alias_count

    # Test with multiple targets (should be ignored)
    parser = Parser()
    initial_alias_count = len(parser.alias)
    node = Assign(
        targets=[
            Name(id="first", ctx=Store()),
            Name(id="second", ctx=Store())
        ],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert len(parser.alias) == initial_alias_count


# LLM-generated content at query #13
#--------------------------

def test_Parser_func_ann():
    # Setup
    parser = Parser()
    root = "test_module"
    
    # Test case 1: No self parameter, no annotations
    args1 = [
        arg(arg="arg1", annotation=None),
        arg(arg="arg2", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result1 = list(parser.func_ann(root, args1, has_self=False, cls_method=False))
    assert result1 == ["Any", "Any", "Any"]
    
    # Test case 2: With self parameter, no annotations
    args2 = [
        arg(arg="self", annotation=None),
        arg(arg="arg1", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result2 = list(parser.func_ann(root, args2, has_self=True, cls_method=False))
    assert result2 == ["Self", "Any", "Any"]
    
    # Test case 3: With self parameter and class method
    args3 = [
        arg(arg="cls", annotation=None),
        arg(arg="arg1", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result3 = list(parser.func_ann(root, args3, has_self=True, cls_method=True))
    assert result3 == ["type[Self]", "Any", "Any"]
    
    # Test case 4: With annotations
    args4 = [
        arg(arg="self", annotation=Name(id="int", ctx=Load())),
        arg(arg="arg1", annotation=Name(id="str", ctx=Load())),
        arg(arg="return", annotation=Name(id="bool", ctx=Load()))
    ]
    result4 = list(parser.func_ann(root, args4, has_self=True, cls_method=False))
    assert result4 == ["Self", "str", "bool"]
    
    # Test case 5: With varargs separator
    args5 = [
        arg(arg="self", annotation=None),
        arg(arg="*", annotation=None),
        arg(arg="arg1", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result5 = list(parser.func_ann(root, args5, has_self=True, cls_method=False))
    assert result5 == ["Self", "", "Any", "Any"]
    
    # Test case 6: With self type annotation
    args6 = [
        arg(arg="self", annotation=Name(id="MyClass", ctx=Load())),
        arg(arg="arg1", annotation=None),
        arg(arg="return", annotation=None)
    ]
    result6 = list(parser.func_ann(root, args6, has_self=True, cls_method=False))
    assert result6 == ["Self", "Any", "Any"]


# LLM-generated content at query #14
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"
    
    # Test Import
    import_node = Import(names=[alias(name='os', asname=None), alias(name='sys', asname='system')])
    parser.imports(root, import_node)
    assert parser.alias == {
        _m(root, 'os'): 'os',
        _m(root, 'system'): 'sys'
    }
    
    # Test ImportFrom
    import_from_node = ImportFrom(module='math', names=[alias(name='sqrt', asname='square_root')], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias == {
        _m(root, 'os'): 'os',
        _m(root, 'system'): 'sys',
        _m(root, 'square_root'): 'math.sqrt'
    }
    
    # Test ImportFrom with level
    import_from_node_with_level = ImportFrom(module='numpy', names=[alias(name='array', asname=None)], level=1)
    parser.imports(root, import_from_node_with_level)
    assert parser.alias == {
        _m(root, 'os'): 'os',
        _m(root, 'system'): 'sys',
        _m(root, 'square_root'): 'math.sqrt',
        _m(root, 'array'): 'test_module.numpy.array'
    }


# LLM-generated content at query #15
#--------------------------

def test_Resolver_visit_Subscript():
    # Test Union case
    resolver = Resolver("root", {})
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(
            elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())],
            ctx=Load()
        ),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name) and result.right.id == "str"

    # Test Optional case
    resolver = Resolver("root", {})
    optional_node = Subscript(
        value=Name(id="Optional", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant) and result.right.value is None

    # Test PEP585 case
    resolver = Resolver("root", {})
    pep585_node = Subscript(
        value=Name(id="List", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(pep585_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name) and result.value.id == "list"

    # Test regular case (no transformation)
    resolver = Resolver("root", {})
    regular_node = Subscript(
        value=Name(id="Custom", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(regular_node)
    assert result is regular_node

    # Test with alias mapping
    resolver = Resolver("root", {"root.Union": "typing.Union"})
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(
            elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())],
            ctx=Load()
        ),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name) and result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name) and result.right.id == "str"


# LLM-generated content at query #16
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test single node
    node = Expr(value=Constant(value="test"))
    assert list(walk_body([node])) == [node]

    # Test If node
    if_node = If(
        test=Constant(value=True),
        body=[Expr(value=Constant(value="if_true"))],
        orelse=[Expr(value=Constant(value="if_false"))]
    )
    expected = [Expr(value=Constant(value="if_true")), Expr(value=Constant(value="if_false"))]
    assert list(walk_body([if_node])) == expected

    # Test Try node
    try_node = Try(
        body=[Expr(value=Constant(value="try_body"))],
        handlers=[],  # No handlers for simplicity
        orelse=[Expr(value=Constant(value="try_orelse"))],
        finalbody=[Expr(value=Constant(value="try_finalbody"))]
    )
    expected = [
        Expr(value=Constant(value="try_body")),
        Expr(value=Constant(value="try_orelse")),
        Expr(value=Constant(value="try_finalbody"))
    ]
    assert list(walk_body([try_node])) == expected

    # Test mixed nodes
    mixed_body = [
        Expr(value=Constant(value="first")),
        if_node,
        try_node,
        Expr(value=Constant(value="last"))
    ]
    expected = [
        Expr(value=Constant(value="first")),
        Expr(value=Constant(value="if_true")),
        Expr(value=Constant(value="if_false")),
        Expr(value=Constant(value="try_body")),
        Expr(value=Constant(value="try_orelse")),
        Expr(value=Constant(value="try_finalbody")),
        Expr(value=Constant(value="last"))
    ]
    assert list(walk_body(mixed_body)) == expected


# LLM-generated content at query #17
#--------------------------

def test_Resolver_visit_Name():
    # Test with self_ty
    resolver = Resolver("root", {}, "SelfType")
    node = Name("SelfType", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test with alias
    resolver = Resolver("root", {"root.name": "other_name"}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "other_name"

    # Test with alias that points to itself (no change)
    resolver = Resolver("root", {"root.name": "root.name"}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"

    # Test with TypeVar in alias (should not change)
    resolver = Resolver("root", {"root.TypeVar": "typing.TypeVar"}, "")
    node = Name("TypeVar", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"

    # Test with no alias and no self_ty
    resolver = Resolver("root", {}, "")
    node = Name("name", Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "name"


# LLM-generated content at query #18
#--------------------------

def test_Resolver_visit_Constant():
    """Test Resolver.visit_Constant method."""
    resolver = Resolver(root="test", alias={}, self_ty="")
    
    # Test with non-string constant
    node = Constant(value=42)
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string that is not a valid name
    node = Constant(value="not a valid name")
    result = resolver.visit_Constant(node)
    assert result == node
    
    # Test with string that is a valid name
    node = Constant(value="valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"
    assert isinstance(result.ctx, Load)
    
    # Test with string that contains syntax error
    node = Constant(value="invalid name with space")
    result = resolver.visit_Constant(node)
    assert result == node


# LLM-generated content at query #19
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    
    # Test AnnAssign with Name target and non-None value
    ann_assign_node = AnnAssign(target=Name(id='x', ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    parser.globals('root', ann_assign_node)
    assert parser.alias['root.x'] == '42'
    
    # Test Assign with single Name target and no type comment
    assign_node = Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=100))
    parser.globals('root', assign_node)
    assert parser.alias['root.y'] == '100'
    assert parser.const['root.y'] == 'int'
    
    # Test Assign with single Name target and type comment
    assign_node_with_type = Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=3.14), type_comment='float')
    parser.globals('root', assign_node_with_type)
    assert parser.alias['root.z'] == '3.14'
    assert parser.const['root.z'] == 'float'
    
    # Test Assign with __all__ and Tuple value
    tuple_node = Tuple(elts=[Constant(value='a'), Constant(value='b')], ctx=Load())
    assign_node_all = Assign(targets=[Name(id='__all__', ctx=Store())], value=tuple_node)
    parser.globals('root', assign_node_all)
    assert 'root.a' in parser.imp['root']
    assert 'root.b' in parser.imp['root']
    
    # Test Assign with non-single target
    multi_target_node = Assign(targets=[Name(id='a', ctx=Store()), Name(id='b', ctx=Store())], value=Constant(value=1))
    parser.globals('root', multi_target_node)
    assert 'root.a' not in parser.alias
    assert 'root.b' not in parser.alias
    
    # Test Assign with non-Name target
    subscript_target_node = Assign(targets=[Subscript(value=Name(id='x', ctx=Load()), slice=Constant(value=0), ctx=Store())], value=Constant(value=1))
    parser.globals('root', subscript_target_node)
    assert 'root.x' not in parser.alias
    
    # Test AnnAssign with non-Name target
    ann_assign_non_name = AnnAssign(target=Subscript(value=Name(id='x', ctx=Load()), slice=Constant(value=0), ctx=Store()), annotation=Name(id='int', ctx=Load()), value=Constant(value=42))
    parser.globals('root', ann_assign_non_name)
    assert 'root.x' not in parser.alias
    
    # Test Assign with non-Tuple/List value for __all__
    assign_node_non_tuple = Assign(targets=[Name(id='__all__', ctx=Store())], value=Constant(value='a'))
    parser.globals('root', assign_node_non_tuple)
    assert 'root.a' not in parser.imp['root']


# LLM-generated content at query #20
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Constant("object")]
    body = [
        AnnAssign(Name("attr1"), annotation=Name("int")),
        Assign([Name("attr2")], Constant(42)),
        Delete([Name("attr2")])
    ]

    parser.class_api(root, name, bases, body)

    assert name in parser.doc
    assert parser.doc[name] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    assert name in parser.level
    assert parser.level[name] == 0
    assert name in parser.root
    assert parser.root[name] == root

    # Test with enum bases
    enum_bases = [Constant("enum.Enum")]
    parser.class_api(root, name, enum_bases, body)
    assert "Enums" in parser.doc[name]

    # Test with members
    member_body = [
        AnnAssign(Name("attr3"), annotation=Name("str")),
        Assign([Name("attr4")], Constant("value"))
    ]
    parser.class_api(root, name, bases, member_body)
    assert "Members" in parser.doc[name]


# LLM-generated content at query #21
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    parser.root['test_module'] = 'test_module'
    parser.alias['test_module'] = 'test_module'
    
    # Mocking the bases and body for the class
    bases = [Constant('BaseClass')]
    body = [
        AnnAssign(Name('attr1'), Constant('int'), Constant(1)),
        Assign([Name('attr2')], Constant('str')),
        Delete([Name('attr1')])
    ]
    
    # Call the method
    parser.class_api('test_module', 'test_module.ClassA', bases, body)
    
    # Check the base class table
    assert "Bases" in parser.doc['test_module.ClassA']
    assert "BaseClass" in parser.doc['test_module.ClassA']
    
    # Check the members table
    assert "Members" in parser.doc['test_module.ClassA']
    assert "attr2" in parser.doc['test_module.ClassA']
    assert "str" in parser.doc['test_module.ClassA']
    
    # Check the deleted attribute is not in the members table
    assert "attr1" not in parser.doc['test_module.ClassA']


# LLM-generated content at query #22
#--------------------------

def test_Resolver_visit_Attribute():
    resolver = Resolver(root="test", alias={})
    
    # Test case 1: Attribute with non-typing value
    node = Attribute(value=Name(id="non_typing", ctx=Load()), attr="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result == node
    
    # Test case 2: Attribute with typing value
    node = Attribute(value=Name(id="typing", ctx=Load()), attr="attr", ctx=Load())
    expected = Name(id="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == expected.id
    assert result.ctx == expected.ctx
    
    # Test case 3: Attribute with non-typing value and nested attribute
    node = Attribute(value=Attribute(value=Name(id="non_typing", ctx=Load()), attr="nested", ctx=Load()), attr="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert result == node
    
    # Test case 4: Attribute with typing value and nested attribute
    node = Attribute(value=Attribute(value=Name(id="typing", ctx=Load()), attr="nested", ctx=Load()), attr="attr", ctx=Load())
    result = resolver.visit_Attribute(node)
    assert isinstance(result, Name)
    assert result.id == "attr"
    assert result.ctx == Load()


# LLM-generated content at query #23
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "example_module"

    # Test AnnAssign with Name target
    ann_assign = AnnAssign(
        target=Name(id="SOME_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert parser.alias[_m(root, "SOME_CONST")] == "42"
    assert parser.const[_m(root, "SOME_CONST")] == "int"

    # Test Assign with Name target
    assign = Assign(
        targets=[Name(id="ANOTHER_CONST", ctx=Store())],
        value=Constant(value=3.14),
        type_comment="float"
    )
    parser.globals(root, assign)
    assert parser.alias[_m(root, "ANOTHER_CONST")] == "3.14"
    assert parser.const[_m(root, "ANOTHER_CONST")] == "float"

    # Test Assign with __all__ and Tuple value
    assign_all = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(
            elts=[Constant(value="SOME_CONST"), Constant(value="ANOTHER_CONST")],
            ctx=Load()
        )
    )
    parser.globals(root, assign_all)
    assert parser.imp[root] == {_m(root, "SOME_CONST"), _m(root, "ANOTHER_CONST")}

    # Test Assign with non-Tuple value for __all__
    assign_non_tuple = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Constant(value="SOME_CONST")
    )
    parser.globals(root, assign_non_tuple)
    assert parser.imp[root] == {_m(root, "SOME_CONST"), _m(root, "ANOTHER_CONST")}

    # Test Assign with non-Name target
    assign_non_name = Assign(
        targets=[Attribute(value=Name(id="module", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value="value")
    )
    parser.globals(root, assign_non_name)
    assert parser.alias == {
        _m(root, "SOME_CONST"): "42",
        _m(root, "ANOTHER_CONST"): "3.14"
    }
    assert parser.const == {
        _m(root, "SOME_CONST"): "int",
        _m(root, "ANOTHER_CONST"): "float"
    }


# LLM-generated content at query #24
#--------------------------

```python
def test_walk_body():
    # Test with simple body
    body = [Assign(targets=[Name(id='x', ctx=Store())], value=Constant(value=42))]
    result = list(walk_body(body))
    assert result == body

    # Test with If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Store())], value=Constant(value=10))],
        orelse=[Assign(targets=[Name(id='z', ctx=Store())], value=Constant(value=20))]
    )
    body = [if_node]
    result = list(walk_body(body))
    assert result == [if_node.body[0], if_node.orelse[0]]

    # Test with Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='a', ctx=Store())], value=Constant(value=1))],
        handlers=[],
        orelse=[Assign(targets=[Name(id='b', ctx=Store())], value=Constant(value=2))],
        finalbody=[Assign(targets=[Name(id='c', ctx=Store())], value=Constant(value=3))]
    )
    body = [try_node]
    result = list(walk_body(body))
    assert result == [try_node.body[0], try_node.orelse[0], try_node.finalbody[0]]

    # Test with nested If and Try statements
    nested_if = If(
        test=Constant(value=False),
        body=[Assign(targets=[Name(id='d', ctx=Store())], value=Constant(value=4))],
        orelse=[try_node]
    )
    body = [nested_if]
    result = list(walk_body(body))
    assert result == [nested_if.orelse[0].body[0], nested_if.orelse[0].orelse[0], nested_if.orelse[0].finalbody[0]]

    # Test empty body
    body = []
    result = list(walk_body(body))
    assert result == []


# LLM-generated content at query #25
#--------------------------

```python
def test_const_type():
    # Test Constant
    node = Constant(value=42)
    assert const_type(node) == "int"
    
    node = Constant(value=3.14)
    assert const_type(node) == "float"
    
    node = Constant(value="hello")
    assert const_type(node) == "str"
    
    # Test Tuple
    node = Tuple(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "tuple[int, int]"
    
    node = Tuple(elts=[Constant(value="a"), Constant(value="b")], ctx=Load())
    assert const_type(node) == "tuple[str, str]"
    
    node = Tuple(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "tuple[Any, Any]"
    
    # Test List
    node = List(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "list[int, int]"
    
    node = List(elts=[Constant(value="a"), Constant(value="b")], ctx=Load())
    assert const_type(node) == "list[str, str]"
    
    node = List(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "list[Any, Any]"
    
    # Test Set
    node = Set(elts=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "set[int, int]"
    
    node = Set(elts=[Constant(value="a"), Constant(value="b")], ctx=Load())
    assert const_type(node) == "set[str, str]"
    
    node = Set(elts=[Constant(value=1), Constant(value="a")], ctx=Load())
    assert const_type(node) == "set[Any, Any]"
    
    # Test Dict
    node = Dict(keys=[Constant(value=1), Constant(value=2)], values=[Constant(value="a"), Constant(value="b")], ctx=Load())
    assert const_type(node) == "dict[int, str]"
    
    node = Dict(keys=[Constant(value=1), Constant(value="a")], values=[Constant(value="a"), Constant(value="b")], ctx=Load())
    assert const_type(node) == "dict[Any, str]"
    
    node = Dict(keys=[Constant(value="a"), Constant(value="b")], values=[Constant(value=1), Constant(value=2)], ctx=Load())
    assert const_type(node) == "dict[str, int]"
    
    # Test Call
    node = Call(func=Name(id="int", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "int"
    
    node = Call(func=Name(id="str", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "str"
    
    node = Call(func=Name(id="list", ctx=Load()), args=[], keywords=[])
    assert const_type(node) == "list"
    
    # Test Any
    node = BinOp(left=Constant(value=1), op=BitOr(), right=Constant(value=2))
    assert const_type(node) == "Any"
    
    node = Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Load())
    assert const_type(node) == "Any"


# LLM-generated content at query #26
#--------------------------

def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with Name target
    ann_assign = AnnAssign(
        target=Name(id="TEST_CONST", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign)
    assert parser.const["test_module.TEST_CONST"] == "int"
    assert parser.alias["test_module.TEST_CONST"] == "42"

    # Test Assign with Name target
    assign = Assign(
        targets=[Name(id="ANOTHER_CONST", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, assign)
    assert parser.const["test_module.ANOTHER_CONST"] == "str"
    assert parser.alias["test_module.ANOTHER_CONST"] == "'hello'"

    # Test Assign with Tuple value (__all__)
    assign_all = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=Tuple(elts=[Constant(value="TEST_CONST"), Constant(value="ANOTHER_CONST")], ctx=Load())
    )
    parser.globals(root, assign_all)
    assert parser.imp[root] == {"test_module.TEST_CONST", "test_module.ANOTHER_CONST"}

    # Test Assign with non-Tuple value (not __all__)
    assign_non_all = Assign(
        targets=[Name(id="NOT_ALL", ctx=Store())],
        value=Constant(value=123)
    )
    parser.globals(root, assign_non_all)
    assert "test_module.NOT_ALL" not in parser.imp[root]

    # Test Assign with multiple targets (should not be processed)
    assign_multi_targets = Assign(
        targets=[Name(id="A", ctx=Store()), Name(id="B", ctx=Store())],
        value=Constant(value=123)
    )
    parser.globals(root, assign_multi_targets)
    assert "test_module.A" not in parser.alias
    assert "test_module.B" not in parser.alias

    # Test AnnAssign with non-Name target (should not be processed)
    ann_assign_non_name = AnnAssign(
        target=Subscript(value=Name(id="test", ctx=Load()), slice=Constant(value=0), ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, ann_assign_non_name)
    assert "test_module.test[0]" not in parser.alias


# LLM-generated content at query #27
#--------------------------

Here's a unit test for the `is_public_family` function:


# LLM-generated content at query #28
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "example_module"
    name = "example_module.example_function"
    node = arguments(
        posonlyargs=[arg(arg="arg1", annotation=None)],
        args=[arg(arg="arg2", annotation=None)],
        vararg=arg(arg="*args", annotation=None),
        kwonlyargs=[arg(arg="kwarg1", annotation=None)],
        kwarg=arg(arg="**kwargs", annotation=None),
        defaults=[],
        kw_defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = "### example_function()\n\n*Full name:* `example_module.example_function`\n\n"
    expected_doc += "| arg1 | arg2 | *args | kwarg1 | **kwargs | return |\n"
    expected_doc += "|------|------|-------|--------|----------|--------|\n"
    expected_doc += "| Any | Any | Any | Any | Any | Any |\n"

    assert parser.doc[name] == expected_doc


# LLM-generated content at query #29
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Constant("base_class")]
    body = [
        AnnAssign(Name("attr1"), Constant("int"), None, None),
        Assign([Name("attr2")], Constant("str"), None),
        Delete([Name("attr2")]),
        AnnAssign(Name("attr3"), Constant("float"), None, None),
        Assign([Name("attr4")], Constant("bool"), None),
    ]

    parser.class_api(root, name, bases, body)

    assert parser.doc[name] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    assert parser.doc[name] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"

    parser = Parser()
    root = "test_module"
    name = "test_module.TestEnum"
    bases = [Constant("enum.Enum")]
    body = [
        AnnAssign(Name("attr1"), Constant("int"), None, None),
        Assign([Name("attr2")], Constant("str"), None),
        Delete([Name("attr2")]),
        AnnAssign(Name("attr3"), Constant("float"), None, None),
        Assign([Name("attr4")], Constant("bool"), None),
    ]

    parser.class_api(root, name, bases, body)

    assert parser.doc[name] == "### class TestEnum\n\n*Full name:* `test_module.TestEnum`\n\n"
    assert parser.doc[name] == "### class TestEnum\n\n*Full name:* `test_module.TestEnum`\n\n"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_public_family():
    # Test public module name
    assert is_public_family("public.module.name") == True
    
    # Test private module name
    assert is_public_family("_private.module.name") == False
    
    # Test mixed public and private module name
    assert is_public_family("public._private.module") == False
    
    # Test magic method in public module
    assert is_public_family("public.module.__magic__") == True
    
    # Test magic method in private module
    assert is_public_family("_private.module.__magic__") == False
    
    # Test single public module
    assert is_public_family("public") == True
    
    # Test single private module
    assert is_public_family("_private") == False
    
    # Test empty string
    assert is_public_family("") == True
    
    # Test name with multiple dots
    assert is_public_family("public.module.submodule.name") == True
    
    # Test name with multiple dots including private
    assert is_public_family("public.module._submodule.name") == False


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"
    
    # Test AnnAssign with Name target and non-None value
    node = AnnAssign(target=Name(id="var_name", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=42))
    parser.globals(root, node)
    assert _m(root, "var_name") in parser.alias
    assert parser.alias[_m(root, "var_name")] == "42"
    
    # Test Assign with Name target and non-None value
    node = Assign(targets=[Name(id="var_name", ctx=Store())], value=Constant(value=42), type_comment="int")
    parser.globals(root, node)
    assert _m(root, "var_name") in parser.alias
    assert parser.alias[_m(root, "var_name")] == "42"
    assert parser.const.get(_m(root, "var_name"), ANY) == "int"
    
    # Test Assign with Tuple value (__all__)
    node = Assign(targets=[Name(id="__all__", ctx=Store())], value=Tuple(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load()))
    parser.globals(root, node)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test Assign with List value (__all__)
    node = Assign(targets=[Name(id="__all__", ctx=Store())], value=List(elts=[Constant(value="func1"), Constant(value="func2")], ctx=Load()))
    parser.globals(root, node)
    assert _m(root, "func1") in parser.imp[root]
    assert _m(root, "func2") in parser.imp[root]
    
    # Test Assign with non-Name target
    node = Assign(targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())], value=Constant(value=42))
    parser.globals(root, node)
    assert _m(root, "var_name") not in parser.alias
    
    # Test Assign with multiple targets
    node = Assign(targets=[Name(id="var1", ctx=Store()), Name(id="var2", ctx=Store())], value=Constant(value=42))
    parser.globals(root, node)
    assert _m(root, "var1") not in parser.alias
    assert _m(root, "var2") not in parser.alias
    
    # Test AnnAssign with non-Name target
    node = AnnAssign(target=Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store()), annotation=Name(id="int", ctx=Load()), value=Constant(value=42))
    parser.globals(root, node)
    assert _m(root, "var_name") not in parser.alias


# LLM-generated content at query #3
#--------------------------

def test_Parser_compile():
    # Test case 1: Empty parser
    p = Parser()
    assert p.compile() == "\n"

    # Test case 2: Single module with no content
    p = Parser()
    p.doc["module"] = "# Module `module`\n\n"
    p.level["module"] = 0
    p.root["module"] = "module"
    p.imp["module"] = set()
    assert p.compile() == "# Module `module`\n\n\n"

    # Test case 3: Module with constants
    p = Parser()
    p.doc["module"] = "# Module `module`\n\n"
    p.level["module"] = 0
    p.root["module"] = "module"
    p.imp["module"] = set()
    p.const["module.CONST"] = "int"
    p.root["module.CONST"] = "module"
    assert p.compile() == "# Module `module`\n\n| Constants | Type |\n|-----------|------|\n| `CONST` | `int` |\n\n\n"

    # Test case 4: Module with function
    p = Parser()
    p.doc["module"] = "# Module `module`\n\n"
    p.doc["module.func"] = "## func()\n\n*Full name:* `module.func`\n\n"
    p.level["module"] = 0
    p.level["module.func"] = 0
    p.root["module"] = "module"
    p.root["module.func"] = "module"
    p.imp["module"] = set()
    assert p.compile() == "# Module `module`\n\n\n## func()\n\n*Full name:* `module.func`\n\n\n"

    # Test case 5: Module with TOC enabled
    p = Parser(toc=True)
    p.doc["module"] = "# Module `module`\n\n"
    p.level["module"] = 0
    p.root["module"] = "module"
    p.imp["module"] = set()
    assert p.compile() == "**Table of contents:**\n+ [`module`](#module)\n\n# Module `module`\n\n\n"

    # Test case 6: Non-public items should be filtered out
    p = Parser()
    p.doc["module"] = "# Module `module`\n\n"
    p.doc["_private"] = "## _private()\n\n*Full name:* `_private`\n\n"
    p.level["module"] = 0
    p.level["_private"] = 0
    p.root["module"] = "module"
    p.root["_private"] = "module"
    p.imp["module"] = set()
    assert p.compile() == "# Module `module`\n\n\n"

    # Test case 7: Module with __all__ filtering
    p = Parser()
    p.doc["module"] = "# Module `module`\n\n"
    p.doc["module.public"] = "## public()\n\n*Full name:* `module.public`\n\n"
    p.doc["module._private"] = "## _private()\n\n*Full name:* `module._private`\n\n"
    p.level["module"] = 0
    p.level["module.public"] = 0
    p.level["module._private"] = 0
    p.root["module"] = "module"
    p.root["module.public"] = "module"
    p.root["module._private"] = "module"
    p.imp["module"] = {"module.public"}
    assert p.compile() == "# Module `module`\n\n\n## public()\n\n*Full name:* `module.public`\n\n\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_Parser_api():
    parser = Parser()
    root = "test_module"
    script = """
    def test_function():
        pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.test_function"] == "### test_function()\n\n*Full name:* `test_module.test_function`\n\n"
    
    script = """
    class TestClass:
        def test_method(self):
            pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.TestClass.test_method"] == "#### test_method()\n\n*Full name:* `test_module.TestClass.test_method`\n\n"
    
    script = """
    @staticmethod
    def test_static_method():
        pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.test_static_method"] == "### test_static_method()\n\n*Full name:* `test_module.test_static_method`\n\n"
    
    script = """
    @classmethod
    def test_class_method(cls):
        pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.test_class_method"] == "### test_class_method()\n\n*Full name:* `test_module.test_class_method`\n\n"
    
    script = """
    class TestClass:
        @staticmethod
        def test_static_method():
            pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.TestClass.test_static_method"] == "#### test_static_method()\n\n*Full name:* `test_module.TestClass.test_static_method`\n\n"
    
    script = """
    class TestClass:
        @classmethod
        def test_class_method(cls):
            pass
    """
    parser.parse(root, script)
    assert parser.doc[f"{root}.TestClass.test_class_method"] == "#### test_class_method()\n\n*Full name:* `test_module.TestClass.test_class_method`\n\n"


# LLM-generated content at query #5
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver(root="example", alias={})
    
    # Test PEP585
    subscript_node = Subscript(value=Name(id="List", ctx=Load()), slice=Name(id="int", ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(subscript_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    
    # Test PEP604 Union
    subscript_node = Subscript(value=Name(id="Union", ctx=Load()), slice=Tuple(elts=[Name(id="int", ctx=Load()), Name(id="str", ctx=Load())], ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(subscript_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    
    # Test PEP604 Optional
    subscript_node = Subscript(value=Name(id="Optional", ctx=Load()), slice=Name(id="int", ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(subscript_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test non-PEP585/604
    subscript_node = Subscript(value=Name(id="Dict", ctx=Load()), slice=Tuple(elts=[Name(id="str", ctx=Load()), Name(id="int", ctx=Load())], ctx=Load()), ctx=Load())
    result = resolver.visit_Subscript(subscript_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"


# LLM-generated content at query #6
#--------------------------

```python
def test_Resolver_visit_Subscript():
    resolver = Resolver(root="test", alias={}, self_ty="Self")
    
    # Test Union type handling
    union_node = Subscript(
        value=Name(id="Union", ctx=Load()),
        slice=Tuple(elts=[
            Name(id="int", ctx=Load()),
            Name(id="str", ctx=Load())
        ], ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"
    
    # Test Optional type handling
    optional_node = Subscript(
        value=Name(id="Optional", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None
    
    # Test PEP585 deprecated type handling
    pep585_node = Subscript(
        value=Name(id="List", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(pep585_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"
    
    # Test non-standard type handling
    non_std_node = Subscript(
        value=Name(id="Custom", ctx=Load()),
        slice=Name(id="int", ctx=Load()),
        ctx=Load()
    )
    result = resolver.visit_Subscript(non_std_node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Custom"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Constant("object")]
    body = [
        AnnAssign(Name("attr"), Constant("int"), None, None),
        Assign([Name("attr2")], Constant("str")),
        Delete([Name("attr")]),
        FunctionDef("method", arguments(), [], [], None, [], None),
    ]

    parser.class_api(root, name, bases, body)

    assert name in parser.doc
    assert "Bases" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "attr" not in parser.doc[name]


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_is_public():
    parser = Parser()
    parser.imp = {
        "module": {"module.public_func", "module.public_class", "module.submodule.public_func"},
        "module.submodule": {"module.submodule.public_func"}
    }
    parser.root = {
        "module": "module",
        "module.public_func": "module",
        "module.public_class": "module",
        "module.submodule": "module.submodule",
        "module.submodule.public_func": "module.submodule"
    }

    # Test public names
    assert parser.is_public("module.public_func") is True
    assert parser.is_public("module.public_class") is True
    assert parser.is_public("module.submodule.public_func") is True

    # Test non-public names
    assert parser.is_public("module.private_func") is False
    assert parser.is_public("module.submodule.private_func") is False

    # Test root module
    assert parser.is_public("module") is True

    # Test submodule
    assert parser.is_public("module.submodule") is True

    # Test non-existent module
    assert parser.is_public("non_existent_module") is False

    # Test with empty __all__
    parser.imp = {
        "module": set(),
        "module.submodule": set()
    }
    assert parser.is_public("module.public_func") is False
    assert parser.is_public("module.submodule.public_func") is False
    assert parser.is_public("module") is True
    assert parser.is_public("module.submodule") is True


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_api():
    parser = Parser()
    root = "test_module"
    node = FunctionDef(name="test_func", args=arguments(), body=[], decorator_list=[], returns=None)
    
    parser.api(root, node)
    
    assert "test_module.test_func" in parser.doc
    assert parser.doc["test_module.test_func"] == "### test_func()\n\n*Full name:* `test_module.test_func`\n\n"
    
    node = AsyncFunctionDef(name="test_async_func", args=arguments(), body=[], decorator_list=[], returns=None)
    
    parser.api(root, node)
    
    assert "test_module.test_async_func" in parser.doc
    assert parser.doc["test_module.test_async_func"] == "### async test_async_func()\n\n*Full name:* `test_module.test_async_func`\n\n"
    
    node = ClassDef(name="TestClass", bases=[], body=[], decorator_list=[])
    
    parser.api(root, node)
    
    assert "test_module.TestClass" in parser.doc
    assert parser.doc["test_module.TestClass"] == "### class TestClass\n\n*Full name:* `test_module.TestClass`\n\n"
    
    node = FunctionDef(name="test_method", args=arguments(), body=[], decorator_list=[], returns=None)
    
    parser.api(root, node, prefix="TestClass")
    
    assert "test_module.TestClass.test_method" in parser.doc
    assert parser.doc["test_module.TestClass.test_method"] == "#### test_method()\n\n*Full name:* `test_module.TestClass.test_method`\n\n"


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_compile():
    parser = Parser()
    parser.doc = {
        "module.func": "# func()\n\n*Full name:* `module.func`\n\n",
        "module.Class": "# class Class\n\n*Full name:* `module.Class`\n\n",
    }
    parser.docstring = {
        "module.func": "Function documentation.",
        "module.Class": "Class documentation.",
    }
    parser.level = {
        "module.func": 1,
        "module.Class": 1,
    }
    parser.root = {
        "module.func": "module",
        "module.Class": "module",
    }
    parser.imp = {
        "module": set(),
    }
    parser.const = {}
    parser.toc = False

    expected_output = (
        "# func()\n\n*Full name:* `module.func`\n\nFunction documentation.\n\n"
        "# class Class\n\n*Full name:* `module.Class`\n\nClass documentation.\n\n"
    )
    assert parser.compile() == expected_output

    parser.toc = True
    expected_output_with_toc = (
        "**Table of contents:**\n"
        "    + [`module.func`](#module-func)\n"
        "    + [`module.Class`](#module-class)\n\n"
        "# func()\n\n*Full name:* `module.func`\n\nFunction documentation.\n\n"
        "# class Class\n\n*Full name:* `module.Class`\n\nClass documentation.\n\n"
    )
    assert parser.compile() == expected_output_with_toc


# LLM-generated content at query #11
#--------------------------

```python
def test_Resolver_visit_Name():
    # Test case 1: Replace Name with Self when node.id matches self_ty
    resolver = Resolver(root="root", alias={}, self_ty="SelfType")
    node = Name(id="SelfType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "Self"

    # Test case 2: Replace Name with its expression from alias
    resolver = Resolver(root="root", alias={"root.SomeType": "SomeExpression"}, self_ty="")
    node = Name(id="SomeType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Expr)
    assert unparse(result) == "SomeExpression"

    # Test case 3: Do not replace Name if it doesn't match self_ty or alias
    resolver = Resolver(root="root", alias={}, self_ty="")
    node = Name(id="AnotherType", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "AnotherType"

    # Test case 4: Handle TypeVar in alias
    resolver = Resolver(root="root", alias={"root.TypeVar": "typing.TypeVar"}, self_ty="")
    node = Name(id="TypeVar", ctx=Load())
    result = resolver.visit_Name(node)
    assert isinstance(result, Name)
    assert result.id == "TypeVar"


# LLM-generated content at query #12
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.func"
    node = arguments(
        posonlyargs=[],
        args=[arg(arg="arg1", annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False

    parser.func_api(root, name, node, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = "### func()\n\n*Full name:* `test_module.func`\n\n| arg1 |\n|------|\n| Any  |\n"
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #13
#--------------------------

def test_Resolver_visit_Constant():
    resolver = Resolver("test_module", {})
    
    # Test with non-string constant
    int_constant = Constant(42)
    result = resolver.visit_Constant(int_constant)
    assert result == int_constant
    
    # Test with string that is not a valid name
    invalid_name = Constant("not a valid name")
    result = resolver.visit_Constant(invalid_name)
    assert result == invalid_name
    
    # Test with string that is a valid name
    valid_name = Constant("valid_name")
    result = resolver.visit_Constant(valid_name)
    assert isinstance(result, Name)
    assert result.id == "valid_name"
    assert isinstance(result.ctx, Load)
    
    # Test with string that is a valid expression
    expr_string = Constant("a + b")
    result = resolver.visit_Constant(expr_string)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "a"
    assert isinstance(result.op, type)
    assert isinstance(result.right, Name)
    assert result.right.id == "b"


# LLM-generated content at query #14
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    
    # Test Import node
    import_node = Import(names=[alias(name='math', asname='m')])
    parser.imports('root', import_node)
    assert parser.alias['root.m'] == 'math'
    
    # Test ImportFrom node with level
    import_from_node = ImportFrom(module='os', names=[alias(name='path', asname=None)], level=1)
    parser.imports('root.module', import_from_node)
    assert parser.alias['root.module.path'] == 'root.os.path'
    
    # Test ImportFrom node without level
    import_from_node_no_level = ImportFrom(module='sys', names=[alias(name='version', asname='ver')], level=0)
    parser.imports('root.module', import_from_node_no_level)
    assert parser.alias['root.module.ver'] == 'sys.version'
    
    # Test Import with multiple names
    import_node_multiple = Import(names=[alias(name='math', asname=None), alias(name='os', asname='o')])
    parser.imports('root', import_node_multiple)
    assert parser.alias['root.math'] == 'math'
    assert parser.alias['root.o'] == 'os'


# LLM-generated content at query #15
#--------------------------

```python
from unittest.mock import patch, MagicMock

def test_Parser_load_docstring():
    parser = Parser()
    parser.doc = {'module.name': '# Module `module.name`\n\n'}
    parser.docstring = {}
    
    mock_module = MagicMock()
    mock_doc = "Mock docstring"
    mock_module.__doc__ = mock_doc
    
    with patch('builtins.getdoc', return_value=mock_doc):
        parser.load_docstring('module', mock_module)
    
    assert parser.docstring['module.name'] == mock_doc


# LLM-generated content at query #16
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Mock alias and root
    alias = {'typing.Union': 'Union', 'typing.Optional': 'Optional', 'typing.List': 'list'}
    root = 'typing'
    
    # Test Union type conversion
    resolver = Resolver(root, alias)
    union_node = Subscript(Name('Union', Load()), Tuple([Name('int', Load()), Name('str', Load())], Load()), Load())
    result = resolver.visit_Subscript(union_node)
    assert isinstance(result, BinOp)
    assert unparse(result) == 'int | str'
    
    # Test Optional type conversion
    resolver = Resolver(root, alias)
    optional_node = Subscript(Name('Optional', Load()), Name('int', Load()), Load())
    result = resolver.visit_Subscript(optional_node)
    assert isinstance(result, BinOp)
    assert unparse(result) == 'int | None'
    
    # Test PEP585 deprecated type conversion
    resolver = Resolver(root, alias)
    list_node = Subscript(Name('List', Load()), Name('int', Load()), Load())
    result = resolver.visit_Subscript(list_node)
    assert isinstance(result, Subscript)
    assert unparse(result) == 'list[int]'
    
    # Test unsupported type remains unchanged
    resolver = Resolver(root, alias)
    unsupported_node = Subscript(Name('Dict', Load()), Tuple([Name('str', Load()), Name('int', Load())], Load()), Load())
    result = resolver.visit_Subscript(unsupported_node)
    assert unparse(result) == 'Dict[str, int]'


# LLM-generated content at query #17
#--------------------------

```python
def test_Parser_globals():
    # Test with AnnAssign (type annotated assignment)
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="TEST_VAR", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.TEST_VAR"] == "42"
    assert parser.const["test_module.TEST_VAR"] == "int"

    # Test with Assign (regular assignment)
    parser = Parser()
    node = Assign(
        targets=[Name(id="ANOTHER_VAR", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.ANOTHER_VAR"] == "'hello'"
    assert parser.const["test_module.ANOTHER_VAR"] == "str"

    # Test with Assign without type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="NO_TYPE_VAR", ctx=Store())],
        value=Constant(value=3.14)
    )
    parser.globals(root, node)
    assert parser.alias["test_module.NO_TYPE_VAR"] == "3.14"
    assert parser.const["test_module.NO_TYPE_VAR"] == "float"

    # Test with __all__ assignment
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[
            Constant(value="public_func"),
            Constant(value="PublicClass")
        ], ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.public_func" in parser.imp[root]
    assert "test_module.PublicClass" in parser.imp[root]

    # Test with non-simple assignment (multiple targets)
    parser = Parser()
    node = Assign(
        targets=[
            Name(id="var1", ctx=Store()),
            Name(id="var2", ctx=Store())
        ],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert "test_module.var1" not in parser.alias
    assert "test_module.var2" not in parser.alias

    # Test with non-Name target
    parser = Parser()
    node = Assign(
        targets=[Attribute(value=Name(id="obj", ctx=Load()), attr="attr", ctx=Store())],
        value=Constant(value=1)
    )
    parser.globals(root, node)
    assert "test_module.obj.attr" not in parser.alias


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_imports():
    # Test Import
    parser = Parser()
    root = "test_module"
    import_node = Import(names=[alias(name='module_name', asname='alias_name')])
    parser.imports(root, import_node)
    assert parser.alias == {'test_module.alias_name': 'module_name'}

    # Test ImportFrom with level
    parser = Parser()
    import_from_node = ImportFrom(module='module_name', names=[alias(name='name', asname='alias_name')], level=1)
    parser.imports(root, import_from_node)
    assert parser.alias == {'test_module.alias_name': 'module_name.name'}

    # Test ImportFrom without level
    parser = Parser()
    import_from_node = ImportFrom(module='module_name', names=[alias(name='name', asname='alias_name')], level=0)
    parser.imports(root, import_from_node)
    assert parser.alias == {'test_module.alias_name': 'module_name.name'}

    # Test multiple imports
    parser = Parser()
    import_node = Import(names=[alias(name='module1', asname='alias1'), alias(name='module2', asname='alias2')])
    parser.imports(root, import_node)
    assert parser.alias == {'test_module.alias1': 'module1', 'test_module.alias2': 'module2'}


# LLM-generated content at query #19
#--------------------------

Here's a unit test for the `visit_Constant` method of the `Resolver` class:


# LLM-generated content at query #20
#--------------------------

def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple body with one statement
    simple_stmt = [Expr(value=Constant(value="test"))]
    assert list(walk_body(simple_stmt)) == simple_stmt

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Expr(value=Constant(value="if body"))],
        orelse=[Expr(value=Constant(value="else body"))]
    )
    if_body = [if_node]
    expected = [
        Expr(value=Constant(value="if body")),
        Expr(value=Constant(value="else body"))
    ]
    assert list(walk_body(if_body)) == expected

    # Test Try statement
    try_node = Try(
        body=[Expr(value=Constant(value="try body"))],
        handlers=[arg(body=[Expr(value=Constant(value="handler body"))])],
        orelse=[Expr(value=Constant(value="else body"))],
        finalbody=[Expr(value=Constant(value="final body"))]
    )
    try_body = [try_node]
    expected = [
        Expr(value=Constant(value="try body")),
        Expr(value=Constant(value="handler body")),
        Expr(value=Constant(value="else body")),
        Expr(value=Constant(value="final body"))
    ]
    assert list(walk_body(try_body)) == expected

    # Test nested structures
    nested_if = If(
        test=Constant(value=True),
        body=[
            Expr(value=Constant(value="outer if")),
            If(
                test=Constant(value=False),
                body=[Expr(value=Constant(value="inner if"))],
                orelse=[]
            )
        ],
        orelse=[]
    )
    nested_body = [nested_if]
    expected = [
        Expr(value=Constant(value="outer if")),
        Expr(value=Constant(value="inner if"))
    ]
    assert list(walk_body(nested_body)) == expected

    # Test mixed nodes
    mixed_body = [
        Expr(value=Constant(value="simple")),
        If(
            test=Constant(value=True),
            body=[Expr(value=Constant(value="if body"))],
            orelse=[]
        ),
        Try(
            body=[Expr(value=Constant(value="try body"))],
            handlers=[],
            orelse=[],
            finalbody=[]
        )
    ]
    expected = [
        Expr(value=Constant(value="simple")),
        Expr(value=Constant(value="if body")),
        Expr(value=Constant(value="try body"))
    ]
    assert list(walk_body(mixed_body)) == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_Parser_is_public():
    # Setup
    parser = Parser()
    parser.root = {
        'module': 'module',
        'module.submodule': 'module.submodule',
        'module.submodule.func': 'module.submodule',
        'module.submodule.Class': 'module.submodule',
        'module.submodule.Class.method': 'module.submodule'
    }
    parser.imp = {
        'module': {'module.submodule', 'module.submodule.func'},
        'module.submodule': {'module.submodule.Class'}
    }

    # Test public names
    assert parser.is_public('module') == True
    assert parser.is_public('module.submodule') == True
    assert parser.is_public('module.submodule.func') == True
    assert parser.is_public('module.submodule.Class') == True
    assert parser.is_public('module.submodule.Class.method') == True

    # Test non-public names
    assert parser.is_public('module.private_func') == False
    assert parser.is_public('module.submodule.private_method') == False

    # Test names not in __all__
    parser.imp['module'] = set()
    parser.imp['module.submodule'] = set()
    assert parser.is_public('module.submodule.func') == False
    assert parser.is_public('module.submodule.Class') == False

    # Test names with public family
    parser.root['module.public_func'] = 'module'
    parser.root['module.submodule.public_method'] = 'module.submodule'
    assert parser.is_public('module.public_func') == True
    assert parser.is_public('module.submodule.public_method') == True

    # Test names with mixed case
    parser.root['module.MixedCase'] = 'module'
    parser.root['module.submodule.MixedCaseMethod'] = 'module.submodule'
    assert parser.is_public('module.MixedCase') == True
    assert parser.is_public('module.submodule.MixedCaseMethod') == True


# LLM-generated content at query #22
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    args = [
        arg(arg="self", annotation=None),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="*args", annotation=Name(id="Any", ctx=Load())),
        arg(arg="**kwargs", annotation=Name(id="Dict", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    annotations = list(parser.func_ann("root", args, has_self=True, cls_method=False))
    assert annotations == ["Self", "int", "str", "Any", "Dict", "None"]

    args = [
        arg(arg="cls", annotation=Name(id="type", ctx=Load())),
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="*args", annotation=Name(id="Any", ctx=Load())),
        arg(arg="**kwargs", annotation=Name(id="Dict", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    annotations = list(parser.func_ann("root", args, has_self=True, cls_method=True))
    assert annotations == ["type[Self]", "int", "str", "Any", "Dict", "None"]

    args = [
        arg(arg="x", annotation=Name(id="int", ctx=Load())),
        arg(arg="y", annotation=Name(id="str", ctx=Load())),
        arg(arg="*args", annotation=Name(id="Any", ctx=Load())),
        arg(arg="**kwargs", annotation=Name(id="Dict", ctx=Load())),
        arg(arg="return", annotation=Name(id="None", ctx=Load()))
    ]
    annotations = list(parser.func_ann("root", args, has_self=False, cls_method=False))
    assert annotations == ["int", "str", "Any", "Dict", "None"]


# LLM-generated content at query #23
#--------------------------

def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("root", {})
    node = Constant(42)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is not a valid name
    node = Constant("not a valid name")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a valid name
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with string constant that has syntax error
    node = Constant("invalid name with space")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a valid expression
    node = Constant("a + b")
    result = resolver.visit_Constant(node)
    assert isinstance(result, BinOp)


# LLM-generated content at query #24
#--------------------------

```python
def test_Parser_parse():
    parser = Parser()
    script = """
import os
from typing import List, Dict

def example_function(param: List[str]) -> Dict[str, int]:
    pass
"""
    parser.parse("test_module", script)
    
    assert "test_module" in parser.doc
    assert "test_module.example_function" in parser.doc
    assert "test_module" in parser.level
    assert "test_module.example_function" in parser.level
    assert "test_module" in parser.imp
    assert "os" in parser.alias
    assert "List" in parser.alias
    assert "Dict" in parser.alias
    assert "test_module.example_function" in parser.docstring or "Missing documentation for test_module.example_function" in parser.compile()


# LLM-generated content at query #25
#--------------------------

```python
def test_doctest():
    # Test with no doctest
    doc = "This is a simple string."
    assert doctest(doc) == "This is a simple string."

    # Test with a single line of doctest
    doc = ">>> print('Hello, World!')\nHello, World!"
    expected = "```python\n>>> print('Hello, World!')\nHello, World!\n```"
    assert doctest(doc) == expected

    # Test with multiple lines of doctest
    doc = ">>> print('Hello')\nHello\n>>> print('World!')\nWorld!"
    expected = "```python\n>>> print('Hello')\nHello\n```\n```python\n>>> print('World!')\nWorld!\n```"
    assert doctest(doc) == expected

    # Test with mixed doctest and regular text
    doc = "This is a test.\n>>> print('Hello')\nHello\nThis is another test."
    expected = "This is a test.\n```python\n>>> print('Hello')\nHello\n```\nThis is another test."
    assert doctest(doc) == expected

    # Test with consecutive doctest lines
    doc = ">>> a = 1\n>>> b = 2\n>>> print(a + b)\n3"
    expected = "```python\n>>> a = 1\n>>> b = 2\n>>> print(a + b)\n3\n```"
    assert doctest(doc) == expected

    # Test with empty string
    doc = ""
    assert doctest(doc) == ""

    # Test with only doctest lines
    doc = ">>> print('Hello')\n>>> print('World')"
    expected = "```python\n>>> print('Hello')\n```\n```python\n>>> print('World')\n```"
    assert doctest(doc) == expected


