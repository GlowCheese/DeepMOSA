####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant(42)
        ),
        Delete(targets=[Name("attr3", Del())])
    ]

    parser.class_api(root, name, bases, body)

    assert name in parser.doc
    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "int" in parser.doc[name]  # const_type should infer this
    assert "attr3" not in parser.doc[name]  # deleted attribute


# LLM-generated content at query #2
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("invalid name")
    result = resolver.visit_Constant(node)
    assert result == node

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.name": "int"})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("name")
    result = resolver.visit_Constant(node)
    assert result == node


# LLM-generated content at query #3
#--------------------------

```python
def test_doctest():
    # Test with no doctest
    assert doctest("This is a normal string.") == "This is a normal string."

    # Test with single line doctest
    assert doctest(">>> print('hello')") == "```python\n>>> print('hello')\n```"

    # Test with multi-line doctest
    input_str = """Example function:
>>> def add(a, b):
...     return a + b
>>> add(1, 2)
3"""
    expected = """Example function:


# LLM-generated content at query #4
#--------------------------

```python
def test_walk_body():
    # Test simple body
    body = [Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))]
    assert list(walk_body(body)) == body

    # Test If statement
    if_node = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))]
    )
    body = [if_node]
    assert list(walk_body(body)) == [
        Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2)),
        Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))
    ]

    # Test Try statement
    try_node = Try(
        body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
        handlers=[],
        orelse=[Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5))],
        finalbody=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))]
    )
    body = [try_node]
    assert list(walk_body(body)) == [
        Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4)),
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5)),
        Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))
    ]

    # Test nested structures
    nested_if = If(
        test=Constant(value=True),
        body=[
            Assign(targets=[Name(id='d', ctx=Load())], value=Constant(value=7)),
            If(
                test=Constant(value=False),
                body=[Assign(targets=[Name(id='e', ctx=Load())], value=Constant(value=8))],
                orelse=[]
            )
        ],
        orelse=[]
    )
    body = [nested_if]
    assert list(walk_body(body)) == [
        Assign(targets=[Name(id='d', ctx=Load())], value=Constant(value=7)),
        Assign(targets=[Name(id='e', ctx=Load())], value=Constant(value=8))
    ]

    # Test empty body
    assert list(walk_body([])) == []


# LLM-generated content at query #5
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test AnnAssign with type annotation
    node_ann = AnnAssign(
        target=Name("VAR1", Load()),
        annotation=Name("int", Load()),
        value=Constant(42)
    )
    parser.globals(root, node_ann)
    assert parser.alias["test_module.VAR1"] == "42"
    assert parser.const["test_module.VAR1"] == "int"

    # Test Assign with type comment
    node_assign = Assign(
        targets=[Name("VAR2", Store())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node_assign)
    assert parser.alias["test_module.VAR2"] == "'hello'"
    assert parser.const["test_module.VAR2"] == "str"

    # Test Assign without type comment (should infer type)
    node_assign_no_comment = Assign(
        targets=[Name("VAR3", Store())],
        value=List(elts=[Constant(1), Constant(2)])
    )
    parser.globals(root, node_assign_no_comment)
    assert parser.alias["test_module.VAR3"] == "[1, 2]"
    assert parser.const["test_module.VAR3"] == "list[int]"

    # Test __all__ handling
    node_all = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("VAR1"), Constant("VAR2")])
    )
    parser.globals(root, node_all)
    assert parser.imp[root] == {"test_module.VAR1", "test_module.VAR2"}

    # Test uppercase constant handling
    node_const = AnnAssign(
        target=Name("CONST", Load()),
        annotation=Name("float", Load()),
        value=Constant(3.14)
    )
    parser.globals(root, node_const)
    assert parser.root["test_module.CONST"] == root
    assert parser.const["test_module.CONST"] == "float"

    # Test non-Name target (should not process)
    node_bad = Assign(
        targets=[Tuple(elts=[Name("x", Store()), Name("y", Store())])],
        value=Constant(10)
    )
    parser.globals(root, node_bad)
    assert "test_module.x" not in parser.alias
    assert "test_module.y" not in parser.alias


# LLM-generated content at query #6
#--------------------------

```python
def test_esc_underscore():
    assert esc_underscore("hello_world") == "hello\\_world"
    assert esc_underscore("single_") == "single_"
    assert esc_underscore("no_underscore") == "no_underscore"
    assert esc_underscore("__double__") == "\\_\\_double\\_\\_"
    assert esc_underscore("") == ""


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_func_ann():
    # Test basic function annotation
    parser = Parser()
    args = [
        arg('x', Name('int', Load())),
        arg('y', Name('str', Load())),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args, has_self=False, cls_method=False))
    assert result == ['int', 'str', 'bool']

    # Test with self parameter (instance method)
    args_with_self = [
        arg('self', Name('Class', Load())),
        arg('x', Name('int', Load())),
        arg('return', Name('str', Load()))
    ]
    result = list(parser.func_ann('module', args_with_self, has_self=True, cls_method=False))
    assert result == ['Self', 'int', 'str']

    # Test with self parameter (class method)
    result = list(parser.func_ann('module', args_with_self, has_self=True, cls_method=True))
    assert result == ['type[Self]', 'int', 'str']

    # Test with no annotation
    args_no_ann = [
        arg('x', None),
        arg('y', Name('str', Load())),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args_no_ann, has_self=False, cls_method=False))
    assert result == ['Any', 'str', 'Any']

    # Test with *args and **kwargs
    args_var = [
        arg('x', Name('int', Load())),
        arg('*', None),
        arg('y', Name('str', Load())),
        arg('**', None),
        arg('return', Name('bool', Load()))
    ]
    result = list(parser.func_ann('module', args_var, has_self=False, cls_method=False))
    assert result == ['int', '', 'str', '', 'bool']

    # Test with complex types
    args_complex = [
        arg('x', Subscript(Name('List', Load()), Name('int', Load()), Load())),
        arg('y', Subscript(Name('Dict', Load()), Tuple([Name('str', Load()), Name('int', Load())], Load()), Load())),
        arg('return', None)
    ]
    result = list(parser.func_ann('module', args_complex, has_self=False, cls_method=False))
    assert result == ['list[int]', 'dict[str, int]', 'Any']


# LLM-generated content at query #8
#--------------------------

```python
def test_Parser_imports():
    # Test Import node
    parser = Parser()
    root = "test_module"
    node = Import(names=[alias(name="os", asname=None), alias(name="sys", asname="system")])
    parser.imports(root, node)
    assert parser.alias["test_module.os"] == "os"
    assert parser.alias["test_module.system"] == "sys"

    # Test ImportFrom node with level 0
    parser = Parser()
    node = ImportFrom(module="collections", names=[alias(name="defaultdict", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.defaultdict"] == "collections.defaultdict"

    # Test ImportFrom node with level 1
    parser = Parser()
    node = ImportFrom(module="submodule", names=[alias(name="func", asname="f")], level=1)
    parser.imports(root, node)
    assert parser.alias["test_module.f"] == "test_module.submodule.func"

    # Test ImportFrom node with None module
    parser = Parser()
    node = ImportFrom(module=None, names=[alias(name="func", asname=None)], level=0)
    parser.imports(root, node)
    assert parser.alias["test_module.func"] == "func"


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_func_api():
    # Test case 1: Simple function with no arguments and no return type
    parser = Parser()
    root = "test_module"
    name = "test_module.simple_func"
    args = arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    returns = None
    has_self = False
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| | |
| --- | --- |
| return | Any |"""
    assert parser.doc[name] == expected_doc

    # Test case 2: Function with positional arguments and return type
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_args"
    args = arguments(
        posonlyargs=[arg("arg1", None), arg("arg2", None)],
        args=[arg("arg3", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = Name("int", Load())
    has_self = False
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| arg1 | arg2 | / | arg3 | return |
| --- | --- | --- | --- | --- |
| Any | Any |  | Any | int |"""
    assert parser.doc[name] == expected_doc

    # Test case 3: Function with default values
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_defaults"
    args = arguments(
        posonlyargs=[],
        args=[arg("arg1", None), arg("arg2", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[Constant(1), Constant(2)]
    )
    returns = None
    has_self = False
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| arg1 | arg2 | return |
| --- | --- | --- |
| Any | Any | Any |
| 1 | 2 |  |"""
    assert parser.doc[name] == expected_doc

    # Test case 4: Function with self parameter (method)
    parser = Parser()
    root = "test_module"
    name = "test_module.MyClass.method"
    args = arguments(
        posonlyargs=[],
        args=[arg("self", None), arg("arg1", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = None
    has_self = True
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| self | arg1 | return |
| --- | --- | --- |
| Self | Any | Any |"""
    assert parser.doc[name] == expected_doc

    # Test case 5: Class method
    parser = Parser()
    root = "test_module"
    name = "test_module.MyClass.class_method"
    args = arguments(
        posonlyargs=[],
        args=[arg("cls", None), arg("arg1", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]
    )
    returns = None
    has_self = True
    cls_method = True

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| cls | arg1 | return |
| --- | --- | --- |
| type[Self] | Any | Any |"""
    assert parser.doc[name] == expected_doc

    # Test case 6: Function with *args and **kwargs
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_varargs"
    args = arguments(
        posonlyargs=[],
        args=[arg("arg1", None)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
        vararg=arg("args", None),
        kwarg=arg("kwargs", None)
    )
    returns = None
    has_self = False
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| arg1 | *args | **kwargs | return |
| --- | --- | --- | --- |
| Any |  |  | Any |"""
    assert parser.doc[name] == expected_doc

    # Test case 7: Function with keyword-only arguments
    parser = Parser()
    root = "test_module"
    name = "test_module.func_with_kwonly"
    args = arguments(
        posonlyargs=[],
        args=[arg("arg1", None)],
        kwonlyargs=[arg("kwarg1", None), arg("kwarg2", None)],
        kw_defaults=[Constant(1), Constant(2)],
        defaults=[]
    )
    returns = None
    has_self = False
    cls_method = False

    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=has_self, cls_method=cls_method)

    expected_doc = """| arg1 | * | kwarg1 | kwarg2 | return |
| --- | --- | --- | --- | --- |
| Any |  | Any | Any | Any |
|  |  | 1 | 2 |  |"""
    assert parser.doc[name] == expected_doc


# LLM-generated content at query #10
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is not a valid expression
    node = Constant("invalid expression")
    assert resolver.visit_Constant(node) == node

    # Test with string constant that is a valid expression
    node = Constant("test_var")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "test_var"

    # Test with string constant that is a valid expression with alias
    resolver = Resolver("test", {"test.test_var": "alias_var"})
    node = Constant("test_var")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "alias_var"


# LLM-generated content at query #11
#--------------------------

```python
def test_Parser_is_public():
    # Test case 1: Public name (starts with uppercase or listed in __all__)
    parser = Parser()
    parser.imp = {'module': {'public_name', 'parent.child'}}
    parser.root = {'module.public_name': 'module', 'module.parent.child': 'module'}
    parser.doc = {'module.public_name': '', 'module.parent.child': ''}
    assert parser.is_public('module.public_name') == True
    assert parser.is_public('module.parent.child') == True

    # Test case 2: Private name (starts with underscore)
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private_name': 'module'}
    assert parser.is_public('module._private_name') == False

    # Test case 3: Name not in __all__ and not public style
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._internal': 'module'}
    assert parser.is_public('module._internal') == False

    # Test case 4: Name in __all__ but not public style (should still be public)
    parser = Parser()
    parser.imp = {'module': {'_in_all'}}
    parser.root = {'module._in_all': 'module'}
    assert parser.is_public('module._in_all') == True

    # Test case 5: Child name of a public parent
    parser = Parser()
    parser.imp = {'module': {'parent'}}
    parser.root = {'module.parent.child': 'module'}
    parser.doc = {'module.parent.child': ''}
    assert parser.is_public('module.parent.child') == True

    # Test case 6: Child name of a private parent
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private.parent.child': 'module'}
    assert parser.is_public('module._private.parent.child') == False


# LLM-generated content at query #12
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.TestClass"
    bases = [Name("BaseClass", Load())]
    body = [
        AnnAssign(
            target=Name("attr1", Store()),
            annotation=Name("int", Load()),
            value=None,
            simple=1
        ),
        Assign(
            targets=[Name("attr2", Store())],
            value=Constant(42),
            type_comment="float"
        ),
        Delete(targets=[Name("attr3", Del())])
    ]

    parser.root[root] = root
    parser.level[root] = 0
    parser.alias = {}
    parser.const = {}
    parser.doc[name] = ""
    parser.docstring[name] = ""
    parser.imp[root] = set()

    parser.class_api(root, name, bases, body)

    assert "Bases" in parser.doc[name]
    assert "BaseClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "attr1" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "attr2" in parser.doc[name]
    assert "float" in parser.doc[name]
    assert "attr3" not in parser.doc[name]


# LLM-generated content at query #13
#--------------------------

```python
def test_Parser_class_api():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
class TestClass:
    '''This is a test class.'''
    x: int
    y: str = "hello"
    def __init__(self):
        pass
"""
    parser.parse(root, script)
    name = "test_module.TestClass"
    bases = []
    body = parser.root[root].body[0].body
    parser.class_api(root, name, bases, body)
    assert name in parser.doc
    assert "class TestClass" in parser.doc[name]
    assert "Members" in parser.doc[name]
    assert "x" in parser.doc[name]
    assert "int" in parser.doc[name]
    assert "y" in parser.doc[name]
    assert "str" in parser.doc[name]


# LLM-generated content at query #14
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    body = [Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))]
    assert list(walk_body(body)) == body

    # Test nested If statements
    body = [
        If(
            test=Constant(value=True),
            body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
            orelse=[]
        )
    ]
    assert len(list(walk_body(body))) == 1

    # Test Try-Except with multiple handlers
    body = [
        Try(
            body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
            handlers=[
                {
                    'type': Name(id='Exception', ctx=Load()),
                    'body': [Expr(value=Constant(value='handler1'))]
                },
                {
                    'type': Name(id='ValueError', ctx=Load()),
                    'body': [Expr(value=Constant(value='handler2'))]
                }
            ],
            orelse=[Expr(value=Constant(value='else'))],
            finalbody=[Expr(value=Constant(value='finally'))]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 4
    assert isinstance(result[0], Assign)
    assert isinstance(result[1], Expr)
    assert isinstance(result[2], Expr)
    assert isinstance(result[3], Expr)

    # Test mixed statements
    body = [
        Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=1)),
        If(
            test=Constant(value=True),
            body=[
                Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=2)),
                Try(
                    body=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=3))],
                    handlers=[],
                    orelse=[],
                    finalbody=[]
                )
            ],
            orelse=[]
        )
    ]
    result = list(walk_body(body))
    assert len(result) == 3
    assert all(isinstance(r, (Assign, Try)) for r in result)


# LLM-generated content at query #15
#--------------------------

```python
def test_Resolver_visit_Constant():
    resolver = Resolver("test", {"test.A": "int"})
    # Test with non-string constant
    node = Constant(123)
    assert resolver.visit_Constant(node) == node
    # Test with string constant that is not a name
    node = Constant("not_a_name")
    assert resolver.visit_Constant(node) == node
    # Test with string constant that is a name
    node = Constant("A")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"
    # Test with string constant that is a name but not in alias
    resolver = Resolver("test", {})
    node = Constant("A")
    assert resolver.visit_Constant(node) == node


# LLM-generated content at query #16
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"
    args = [
        arg("x", Name("int", Load())),
        arg("y", Name("str", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]

    args_with_self = [
        arg("self", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("bool", Load()))
    ]
    result_with_self = list(parser.func_ann(root, args_with_self, has_self=True, cls_method=False))
    assert result_with_self == ["Self", "int", "bool"]

    args_with_cls_method = [
        arg("cls", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("bool", Load()))
    ]
    result_with_cls_method = list(parser.func_ann(root, args_with_cls_method, has_self=True, cls_method=True))
    assert result_with_cls_method == ["type[Self]", "int", "bool"]

    args_with_varargs = [
        arg("x", Name("int", Load())),
        arg("*", None),
        arg("y", Name("str", Load())),
        arg("return", Name("bool", Load()))
    ]
    result_with_varargs = list(parser.func_ann(root, args_with_varargs, has_self=False, cls_method=False))
    assert result_with_varargs == ["int", "", "str", "bool"]

    args_with_no_annotation = [
        arg("x", None),
        arg("y", Name("str", Load())),
        arg("return", None)
    ]
    result_with_no_annotation = list(parser.func_ann(root, args_with_no_annotation, has_self=False, cls_method=False))
    assert result_with_no_annotation == ["Any", "str", "Any"]


# LLM-generated content at query #17
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()

    # Test AnnAssign with type annotation
    node = AnnAssign(
        target=Name("x", Load()),
        annotation=Name("int", Load()),
        value=Constant(1)
    )
    parser.globals("test", node)
    assert parser.alias["test.x"] == "1"
    assert parser.const["test.x"] == "int"
    assert parser.root["test.x"] == "test"

    # Test AnnAssign without value (should not add to alias/const)
    node = AnnAssign(
        target=Name("y", Load()),
        annotation=Name("str", Load()),
        value=None
    )
    parser.globals("test", node)
    assert "test.y" not in parser.alias
    assert "test.y" not in parser.const

    # Test Assign with type comment
    node = Assign(
        targets=[Name("z", Store())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals("test", node)
    assert parser.alias["test.z"] == "3.14"
    assert parser.const["test.z"] == "float"
    assert parser.root["test.z"] == "test"

    # Test Assign without type comment (should infer type)
    node = Assign(
        targets=[Name("w", Store())],
        value=List(elts=[Constant(1), Constant(2)])
    )
    parser.globals("test", node)
    assert parser.alias["test.w"] == "[1, 2]"
    assert parser.const["test.w"] == "list[int]"

    # Test __all__ handling
    node = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("public_func"), Constant("PublicClass")])
    )
    parser.globals("test", node)
    assert parser.imp["test"] == {"test.public_func", "test.PublicClass"}

    # Test non-Name target (should not add to alias/const)
    node = Assign(
        targets=[Subscript(Name("arr", Store()), Constant(0), Store())],
        value=Constant(42)
    )
    parser.globals("test", node)
    assert "test.arr" not in parser.alias
    assert "test.arr" not in parser.const

    # Test multiple targets (should not add to alias/const)
    node = Assign(
        targets=[Name("a", Store()), Name("b", Store())],
        value=Constant(10)
    )
    parser.globals("test", node)
    assert "test.a" not in parser.alias
    assert "test.b" not in parser.alias


# LLM-generated content at query #18
#--------------------------

```python
def test_Parser_imports():
    parser = Parser()
    root = "test_module"

    # Test Import node
    import_node = Import(names=[alias(name="os", asname=None)])
    parser.imports(root, import_node)
    assert parser.alias["test_module.os"] == "os"

    import_node_with_as = Import(names=[alias(name="numpy", asname="np")])
    parser.imports(root, import_node_with_as)
    assert parser.alias["test_module.np"] == "numpy"

    # Test ImportFrom node with level 0
    from_node = ImportFrom(module="sys", names=[alias(name="path", asname=None)], level=0)
    parser.imports(root, from_node)
    assert parser.alias["test_module.path"] == "sys.path"

    from_node_with_as = ImportFrom(module="collections", names=[alias(name="defaultdict", asname="dd")], level=0)
    parser.imports(root, from_node_with_as)
    assert parser.alias["test_module.dd"] == "collections.defaultdict"

    # Test ImportFrom node with level > 0
    from_node_with_level = ImportFrom(module="os", names=[alias(name="path", asname=None)], level=1)
    parser.imports(root, from_node_with_level)
    assert parser.alias["test_module.path"] == "test_module.os.path"

    # Test multiple imports
    multi_import = Import(names=[
        alias(name="sys", asname=None),
        alias(name="os", asname="operating_system")
    ])
    parser.imports(root, multi_import)
    assert parser.alias["test_module.sys"] == "sys"
    assert parser.alias["test_module.operating_system"] == "os"


# LLM-generated content at query #19
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"
    name = "test_module.test_func"
    args = arguments(
        posonlyargs=[arg("a", None), arg("b", None)],
        args=[arg("c", None), arg("d", None)],
        defaults=[Constant(1), Constant(2)],
        vararg=None,
        kwonlyargs=[arg("e", None), arg("f", None)],
        kw_defaults=[Constant(3), Constant(4)],
        kwarg=None
    )
    returns = Constant("result")
    parser.doc[name] = ""
    parser.func_api(root, name, args, returns, has_self=False, cls_method=False)
    assert "a" in parser.doc[name]
    assert "b" in parser.doc[name]
    assert "c" in parser.doc[name]
    assert "d" in parser.doc[name]
    assert "e" in parser.doc[name]
    assert "f" in parser.doc[name]
    assert "return" in parser.doc[name]
    assert "1" in parser.doc[name]
    assert "2" in parser.doc[name]
    assert "3" in parser.doc[name]
    assert "4" in parser.doc[name]
    assert "result" in parser.doc[name]


# LLM-generated content at query #20
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()

    # Test case 1: AnnAssign with type annotation
    node_annassign = AnnAssign(
        target=Name(id="x", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=5),
        simple=1
    )
    parser.globals("test_module", node_annassign)
    assert parser.alias["test_module.x"] == "5"
    assert parser.const["test_module.x"] == "int"

    # Test case 2: Assign with type comment
    node_assign = Assign(
        targets=[Name(id="y", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals("test_module", node_assign)
    assert parser.alias["test_module.y"] == "'hello'"
    assert parser.const["test_module.y"] == "str"

    # Test case 3: Assign without type comment (constant type inference)
    node_assign_no_comment = Assign(
        targets=[Name(id="z", ctx=Store())],
        value=List(elts=[Constant(value=1), Constant(value=2)])
    )
    parser.globals("test_module", node_assign_no_comment)
    assert parser.alias["test_module.z"] == "[1, 2]"
    assert parser.const["test_module.z"] == "list[int]"

    # Test case 4: __all__ handling
    node_all = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="public_func"), Constant(value="public_var")])
    )
    parser.globals("test_module", node_all)
    assert parser.imp["test_module"] == {"test_module.public_func", "test_module.public_var"}

    # Test case 5: Non-constant assignment (should not add to const)
    node_non_const = Assign(
        targets=[Name(id="non_const", ctx=Store())],
        value=Call(func=Name(id="some_func", ctx=Load()), args=[])
    )
    parser.globals("test_module", node_non_const)
    assert "test_module.non_const" in parser.alias
    assert "test_module.non_const" not in parser.const

    # Test case 6: Multiple targets (should not process)
    node_multi_target = Assign(
        targets=[Name(id="a", ctx=Store()), Name(id="b", ctx=Store())],
        value=Constant(value=10)
    )
    parser.globals("test_module", node_multi_target)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #21
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: AnnAssign with type annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name("VAR", Load()),
        annotation=Name("int", Load()),
        value=Constant(42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias.get("test_module.VAR") == "42"
    assert parser.const.get("test_module.VAR") == "int"
    assert parser.root.get("test_module.VAR") == root

    # Test case 2: Assign with type comment
    parser = Parser()
    node = Assign(
        targets=[Name("CONST", Load())],
        value=Constant(3.14),
        type_comment="float"
    )
    parser.globals(root, node)
    assert parser.alias.get("test_module.CONST") == "3.14"
    assert parser.const.get("test_module.CONST") == "float"
    assert parser.root.get("test_module.CONST") == root

    # Test case 3: Assign without type comment (constant type inference)
    parser = Parser()
    node = Assign(
        targets=[Name("LIST", Load())],
        value=List(elts=[Constant(1), Constant(2)], ctx=Load())
    )
    parser.globals(root, node)
    assert parser.alias.get("test_module.LIST") == "[1, 2]"
    assert parser.const.get("test_module.LIST") == "list[int]"
    assert parser.root.get("test_module.LIST") == root

    # Test case 4: __all__ handling
    parser = Parser()
    node = Assign(
        targets=[Name("__all__", Load())],
        value=List(elts=[Constant("public_func"), Constant("PublicClass")], ctx=Load())
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.PublicClass"}

    # Test case 5: Non-constant assignment (should not be added to const)
    parser = Parser()
    node = Assign(
        targets=[Name("non_const", Load())],
        value=Name("some_var", Load())
    )
    parser.globals(root, node)
    assert parser.alias.get("test_module.non_const") == "some_var"
    assert "test_module.non_const" not in parser.const

    # Test case 6: Multiple targets (should be ignored)
    parser = Parser()
    node = Assign(
        targets=[Name("a", Load()), Name("b", Load())],
        value=Constant(10)
    )
    parser.globals(root, node)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #22
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test Union case
    resolver = Resolver("test", {"test.typing.Union": "typing.Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Name)
    assert result.right.id == "str"

    # Test Optional case
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.op, BitOr)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test PEP585 case
    resolver = Resolver("test", {"test.typing.List": "typing.List"})
    node = Subscript(Name("List", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "int"

    # Test non-special case
    node = Subscript(Name("SomeClass", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert result is node


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_public_family():
    assert is_public_family("module.submodule")
    assert is_public_family("module.submodule.Class")
    assert is_public_family("module.submodule.Class.method")
    assert is_public_family("module.submodule.Class.__init__")
    assert is_public_family("module.submodule.__class__")
    assert not is_public_family("_private_module")
    assert not is_public_family("module._private_submodule")
    assert not is_public_family("module.submodule._private_class")
    assert not is_public_family("module.submodule.Class._private_method")
    assert not is_public_family("module.submodule.Class.__private_magic__")


# LLM-generated content at query #2
#--------------------------

```python
def test_Parser_globals():
    parser = Parser()
    root = "test_module"

    # Test case 1: AnnAssign with type annotation
    node1 = AnnAssign(
        target=Name("VAR1", Load()),
        annotation=Name("int", Load()),
        value=Constant(42)
    )
    parser.globals(root, node1)
    assert parser.alias["test_module.VAR1"] == "42"
    assert parser.const["test_module.VAR1"] == "int"

    # Test case 2: Assign with type comment
    node2 = Assign(
        targets=[Name("VAR2", Store())],
        value=Constant("hello"),
        type_comment="str"
    )
    parser.globals(root, node2)
    assert parser.alias["test_module.VAR2"] == "'hello'"
    assert parser.const["test_module.VAR2"] == "str"

    # Test case 3: Assign without type comment (constant type inference)
    node3 = Assign(
        targets=[Name("VAR3", Store())],
        value=List(elts=[Constant(1), Constant(2)])
    )
    parser.globals(root, node3)
    assert parser.alias["test_module.VAR3"] == "[1, 2]"
    assert parser.const["test_module.VAR3"] == "list[int]"

    # Test case 4: __all__ handling
    node4 = Assign(
        targets=[Name("__all__", Store())],
        value=List(elts=[Constant("public_func"), Constant("public_var")])
    )
    parser.globals(root, node4)
    assert parser.imp[root] == {"test_module.public_func", "test_module.public_var"}

    # Test case 5: Non-constant assignment (should not add to const)
    node5 = Assign(
        targets=[Name("non_const", Store())],
        value=Call(func=Name("some_func", Load()), args=[])
    )
    parser.globals(root, node5)
    assert "test_module.non_const" in parser.alias
    assert "test_module.non_const" not in parser.const

    # Test case 6: Multiple targets (should not process)
    node6 = Assign(
        targets=[Name("a", Store()), Name("b", Store())],
        value=Constant(10)
    )
    parser.globals(root, node6)
    assert "test_module.a" not in parser.alias
    assert "test_module.b" not in parser.alias


# LLM-generated content at query #3
#--------------------------

```python
def test_Parser_compile():
    # Test basic compilation
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", "def func(): pass")
    result = parser.compile()
    assert "Module `test_module`" in result
    assert "func()" in result

    # Test with docstring
    parser = Parser.new(link=False, level=2, toc=False)
    parser.parse("test_module", '''"""
    This is a test module.
    """''')
    result = parser.compile()
    assert "This is a test module." in result

    # Test with class and method
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", '''
class TestClass:
    """A test class."""
    def method(self):
        """A test method."""
        pass
''')
    result = parser.compile()
    assert "class TestClass" in result
    assert "A test class." in result
    assert "method()" in result
    assert "A test method." in result

    # Test with imports
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", '''
from typing import List
import os

def func() -> List[int]:
    pass
''')
    result = parser.compile()
    assert "func()" in result
    assert "List[int]" in result

    # Test with constants
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", '''
CONSTANT = 42
''')
    result = parser.compile()
    assert "Constants" in result
    assert "CONSTANT" in result
    assert "int" in result

    # Test with private members
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", '''
def _private_func():
    pass

class TestClass:
    def _private_method(self):
        pass
''')
    result = parser.compile()
    assert "_private_func" not in result
    assert "_private_method" not in result

    # Test with __all__
    parser = Parser.new(link=True, level=1, toc=True)
    parser.parse("test_module", '''
__all__ = ['public_func']

def public_func():
    pass

def _private_func():
    pass
''')
    result = parser.compile()
    assert "public_func()" in result
    assert "_private_func" not in result


# LLM-generated content at query #4
#--------------------------

```python
def test_Parser_api():
    # Test case 1: FunctionDef without prefix
    parser = Parser()
    root = "test_module"
    node = FunctionDef(
        name="test_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=Name(id="str", ctx=Load())
    )
    parser.api(root, node)
    expected_doc = """## test_func()

*Full name:* `test_module.test_func`
<a id="test_module-test_func"></a>

|  |   |
|---|---|
| **x** | `int` |
| **return** | `str` |"""
    assert "test_module.test_func" in parser.doc
    assert parser.doc["test_module.test_func"].strip() == expected_doc.strip()

    # Test case 2: AsyncFunctionDef with prefix
    parser = Parser()
    node = AsyncFunctionDef(
        name="async_test",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=Name(id="None", ctx=Load())
    )
    parser.api(root, node, prefix="TestClass")
    expected_doc = """#### async TestClass.async_test()

*Full name:* `test_module.TestClass.async_test`
<a id="test_module-TestClass-async_test"></a>

|  |   |
|---|---|
| **return** | `None` |"""
    assert "test_module.TestClass.async_test" in parser.doc
    assert parser.doc["test_module.TestClass.async_test"].strip() == expected_doc.strip()

    # Test case 3: ClassDef with bases and body
    parser = Parser()
    node = ClassDef(
        name="TestClass",
        bases=[Name(id="BaseClass", ctx=Load())],
        body=[
            AnnAssign(
                target=Name(id="attr1", ctx=Store()),
                annotation=Name(id="int", ctx=Load()),
                value=Constant(value=1),
                simple=1
            ),
            FunctionDef(
                name="method1",
                args=arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[],
                decorator_list=[],
                returns=Name(id="str", ctx=Load())
            )
        ],
        decorator_list=[]
    )
    parser.api(root, node)
    expected_doc = """## class TestClass

*Full name:* `test_module.TestClass`
<a id="test_module-TestClass"></a>

**Bases:** `BaseClass`

| Members | Type |
|---------|------|
| `attr1` | `int` |"""
    assert "test_module.TestClass" in parser.doc
    assert parser.doc["test_module.TestClass"].strip().startswith(expected_doc.strip())

    # Test case 4: FunctionDef with decorators
    parser = Parser()
    node = FunctionDef(
        name="decorated_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[
            Name(id="staticmethod", ctx=Load()),
            Name(id="custom_decorator", ctx=Load())
        ],
        returns=Name(id="None", ctx=Load())
    )
    parser.api(root, node)
    expected_doc = """## decorated_func()

*Full name:* `test_module.decorated_func`
<a id="test_module-decorated_func"></a>

**Decorators:**
|  |   |
|---|---|
| `@staticmethod` |  |
| `@custom_decorator` |  |"""
    assert "test_module.decorated_func" in parser.doc
    assert parser.doc["test_module.decorated_func"].strip().startswith(expected_doc.strip())


# LLM-generated content at query #5
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"test.typing.List": "list"})
    node = Subscript(Name("List", Load()), Tuple([Name("int", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"test.typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Name("int", Load()), Name("str", Load())], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, BinOp)
    assert isinstance(result.right.left, Name)
    assert result.right.left.id == "str"

    # Test Optional conversion
    resolver = Resolver("test", {"test.typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Name("int", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Name)
    assert result.left.id == "int"
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-typing attribute
    resolver = Resolver("test", {})
    node = Subscript(Name("SomeClass", Load()), Name("attr", Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert result.value.id == "SomeClass"
    assert isinstance(result.slice, Name)
    assert result.slice.id == "attr"


# LLM-generated content at query #6
#--------------------------

```python
def test_Parser_load_docstring():
    # Setup
    parser = Parser()
    parser.doc = {
        'test_module': '# Module `test_module`',
        'test_module.TestClass': '## class TestClass',
        'test_module.test_function': '## test_function()'
    }
    parser.docstring = {}

    # Mock module
    class MockModule:
        __doc__ = "Module docstring"
        TestClass = type('TestClass', (), {'__doc__': "Class docstring"})
        test_function = lambda: None
        test_function.__doc__ = "Function docstring"

    # Test
    parser.load_docstring('test_module', MockModule)

    # Assertions
    assert parser.docstring['test_module'] == "Module docstring"
    assert parser.docstring['test_module.TestClass'] == "Class docstring"
    assert parser.docstring['test_module.test_function'] == "Function docstring"


# LLM-generated content at query #7
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"

    # Test case 1: Simple function with no arguments and no return
    node = FunctionDef(
        name="simple_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.simple_func", node.args, node.returns, has_self=False, cls_method=False)
    assert "test_module.simple_func" in parser.doc
    assert "return" in parser.doc["test_module.simple_func"]

    # Test case 2: Function with positional arguments and return type
    node = FunctionDef(
        name="func_with_args",
        args=arguments(
            posonlyargs=[arg(arg="x", annotation=Name(id="int", ctx=Load()))],
            args=[arg(arg="y", annotation=Name(id="str", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=Name(id="bool", ctx=Load()),
        decorator_list=[]
    )
    parser.func_api(root, "test_module.func_with_args", node.args, node.returns, has_self=False, cls_method=False)
    assert "test_module.func_with_args" in parser.doc
    assert "x" in parser.doc["test_module.func_with_args"]
    assert "y" in parser.doc["test_module.func_with_args"]
    assert "return" in parser.doc["test_module.func_with_args"]

    # Test case 3: Function with default values
    node = FunctionDef(
        name="func_with_defaults",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="a", annotation=Name(id="int", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[Constant(value=1)]
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.func_with_defaults", node.args, node.returns, has_self=False, cls_method=False)
    assert "test_module.func_with_defaults" in parser.doc
    assert "a" in parser.doc["test_module.func_with_defaults"]
    assert "1" in parser.doc["test_module.func_with_defaults"]

    # Test case 4: Function with *args and **kwargs
    node = FunctionDef(
        name="func_with_varargs",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=arg(arg="args", annotation=None),
            kwarg=arg(arg="kwargs", annotation=None)
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.func_with_varargs", node.args, node.returns, has_self=False, cls_method=False)
    assert "test_module.func_with_varargs" in parser.doc
    assert "*args" in parser.doc["test_module.func_with_varargs"]
    assert "**kwargs" in parser.doc["test_module.func_with_varargs"]

    # Test case 5: Function with self parameter (method)
    node = FunctionDef(
        name="method",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="self", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=None,
        decorator_list=[]
    )
    parser.func_api(root, "test_module.method", node.args, node.returns, has_self=True, cls_method=False)
    assert "test_module.method" in parser.doc
    assert "Self" in parser.doc["test_module.method"]

    # Test case 6: Class method with cls parameter
    node = FunctionDef(
        name="cls_method",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="cls", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        returns=None,
        decorator_list=[Name(id="classmethod", ctx=Load())]
    )
    parser.func_api(root, "test_module.cls_method", node.args, node.returns, has_self=False, cls_method=True)
    assert "test_module.cls_method" in parser.doc
    assert "type[Self]" in parser.doc["test_module.cls_method"]


# LLM-generated content at query #8
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Expr(value=Constant(value=1))
    stmt2 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=2))
    body = [stmt1, stmt2]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test If node
    if_node = If(
        test=Constant(value=True),
        body=[stmt1],
        orelse=[stmt2]
    )
    body = [if_node]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test Try node
    try_node = Try(
        body=[stmt1],
        handlers=[],
        orelse=[stmt2],
        finalbody=[]
    )
    body = [try_node]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test nested If nodes
    nested_if = If(
        test=Constant(value=True),
        body=[If(
            test=Constant(value=False),
            body=[stmt1],
            orelse=[stmt2]
        )],
        orelse=[]
    )
    body = [nested_if]
    assert list(walk_body(body)) == [stmt1, stmt2]

    # Test Try node with handlers
    handler_stmt = Expr(value=Constant(value=3))
    try_with_handler = Try(
        body=[stmt1],
        handlers=[{
            'type': None,
            'name': None,
            'body': [handler_stmt]
        }],
        orelse=[],
        finalbody=[]
    )
    body = [try_with_handler]
    assert list(walk_body(body)) == [stmt1, handler_stmt]


# LLM-generated content at query #9
#--------------------------

```python
def test_Parser_globals():
    # Test case 1: AnnAssign with type annotation
    parser = Parser()
    root = "test_module"
    node = AnnAssign(
        target=Name(id="test_var", ctx=Store()),
        annotation=Name(id="int", ctx=Load()),
        value=Constant(value=42),
        simple=1
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "42"
    assert parser.const["test_module.test_var"] == "int"

    # Test case 2: Assign with type comment
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=Constant(value="hello"),
        type_comment="str"
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "'hello'"
    assert parser.const["test_module.test_var"] == "str"

    # Test case 3: Assign without type comment (constant type inference)
    parser = Parser()
    node = Assign(
        targets=[Name(id="test_var", ctx=Store())],
        value=List(elts=[Constant(value=1), Constant(value=2)])
    )
    parser.globals(root, node)
    assert parser.alias["test_module.test_var"] == "[1, 2]"
    assert parser.const["test_module.test_var"] == "list[int]"

    # Test case 4: __all__ handling
    parser = Parser()
    node = Assign(
        targets=[Name(id="__all__", ctx=Store())],
        value=List(elts=[Constant(value="public_func"), Constant(value="public_class")])
    )
    parser.globals(root, node)
    assert parser.imp[root] == {"test_module.public_func", "test_module.public_class"}

    # Test case 5: Non-constant assignment (should not add to const)
    parser = Parser()
    node = Assign(
        targets=[Name(id="non_const", ctx=Store())],
        value=Name(id="some_var", ctx=Load())
    )
    parser.globals(root, node)
    assert "test_module.non_const" not in parser.const

    # Test case 6: Multiple targets (should not process)
    parser = Parser()
    node = Assign(
        targets=[Name(id="var1", ctx=Store()), Name(id="var2", ctx=Store())],
        value=Constant(value=10)
    )
    parser.globals(root, node)
    assert "test_module.var1" not in parser.alias
    assert "test_module.var2" not in parser.alias


# LLM-generated content at query #10
#--------------------------

```python
def test_Parser_api():
    # Test case 1: Function with no prefix
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
def test_function():
    '''This is a test function.'''
    pass
"""
    parser.parse(root, script)
    expected_doc = """## test_function()
*Full name:* `test_module.test_function`

"""
    assert parser.doc["test_module.test_function"] == expected_doc

    # Test case 2: Async function with prefix
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
class TestClass:
    async def test_method(self):
        '''This is a test method.'''
        pass
"""
    parser.parse(root, script)
    expected_doc = """### async test_method()
*Full name:* `test_module.TestClass.test_method`

"""
    assert parser.doc["test_module.TestClass.test_method"] == expected_doc

    # Test case 3: Class with decorators
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
@decorator1
@decorator2
class TestClass:
    '''This is a test class.'''
    pass
"""
    parser.parse(root, script)
    expected_doc = """## class TestClass
*Full name:* `test_module.TestClass`

| Decorators |
|------------|
| `@decorator1` |
| `@decorator2` |
"""
    assert parser.doc["test_module.TestClass"] == expected_doc

    # Test case 4: Function with returns annotation
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
def test_function() -> int:
    '''This is a test function.'''
    return 1
"""
    parser.parse(root, script)
    expected_doc = """## test_function()
*Full name:* `test_module.test_function`

|  |  |
|---|---|
| return | `int` |
"""
    assert parser.doc["test_module.test_function"] == expected_doc

    # Test case 5: Class with bases
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
class TestClass(BaseClass):
    '''This is a test class.'''
    pass
"""
    parser.parse(root, script)
    expected_doc = """## class TestClass
*Full name:* `test_module.TestClass`

| Bases |
|-------|
| `BaseClass` |
"""
    assert parser.doc["test_module.TestClass"] == expected_doc

    # Test case 6: Nested class
    parser = Parser.new(link=False, level=1, toc=False)
    root = "test_module"
    script = """
class OuterClass:
    class InnerClass:
        '''This is an inner class.'''
        pass
"""
    parser.parse(root, script)
    expected_doc = """### class InnerClass
*Full name:* `test_module.OuterClass.InnerClass`

"""
    assert parser.doc["test_module.OuterClass.InnerClass"] == expected_doc


# LLM-generated content at query #11
#--------------------------

```python
def test_Parser_is_public():
    # Test case 1: Public name (starts with uppercase or listed in __all__)
    parser = Parser()
    parser.imp = {'module': {'public_name', 'another_name'}}
    parser.root = {'module.public_name': 'module', 'module._private': 'module'}
    parser.doc = {'module.public_name': '', 'module._private': ''}
    assert parser.is_public('module.public_name') == True
    assert parser.is_public('module._private') == False

    # Test case 2: Name listed in __all__
    parser = Parser()
    parser.imp = {'module': {'listed_name'}}
    parser.root = {'module.listed_name': 'module'}
    parser.doc = {'module.listed_name': ''}
    assert parser.is_public('module.listed_name') == True

    # Test case 3: Private name (starts with underscore)
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private': 'module'}
    parser.doc = {'module._private': ''}
    assert parser.is_public('module._private') == False

    # Test case 4: Name in submodule listed in parent's __all__
    parser = Parser()
    parser.imp = {'parent': {'parent.child'}}
    parser.root = {'parent.child.sub': 'parent.child'}
    parser.doc = {'parent.child.sub': ''}
    assert parser.is_public('parent.child.sub') == True

    # Test case 5: Name not in __all__ and not public style
    parser = Parser()
    parser.imp = {'module': set()}
    parser.root = {'module._private_sub': 'module'}
    parser.doc = {'module._private_sub': ''}
    assert parser.is_public('module._private_sub') == False

    # Test case 6: Name is root module
    parser = Parser()
    parser.imp = {'module': {'module'}}
    parser.root = {'module': 'module'}
    parser.doc = {'module': ''}
    assert parser.is_public('module') == True


# LLM-generated content at query #12
#--------------------------

```python
def test_Resolver_visit_Name():
    resolver = Resolver("test", {"test.a": "int", "test.b": "str"})
    # Test with self_ty
    resolver.self_ty = "Self"
    node = Name("Self", Load())
    assert isinstance(resolver.visit_Name(node), Name)
    assert resolver.visit_Name(node).id == "Self"
    # Test with alias replacement
    node = Name("a", Load())
    assert isinstance(resolver.visit_Name(node), Name)
    assert resolver.visit_Name(node).id == "int"
    # Test without replacement
    node = Name("c", Load())
    assert isinstance(resolver.visit_Name(node), Name)
    assert resolver.visit_Name(node).id == "c"


# LLM-generated content at query #13
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == 123

    # Test with string constant that is not a valid name
    resolver = Resolver("test", {})
    node = Constant("invalid name!")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Constant)
    assert result.value == "invalid name!"

    # Test with string constant that is a valid name
    resolver = Resolver("test", {"test.ValidName": "int"})
    node = Constant("ValidName")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "int"

    # Test with string constant that is a valid name but not in alias
    resolver = Resolver("test", {})
    node = Constant("ValidName")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "ValidName"


# LLM-generated content at query #14
#--------------------------

```python
def test_Resolver_visit_Subscript():
    # Test PEP585 conversion
    resolver = Resolver("test", {"typing.List": "list"})
    node = Subscript(Name("List", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "list"

    # Test Union conversion
    resolver = Resolver("test", {"typing.Union": "Union"})
    node = Subscript(Name("Union", Load()), Tuple([Constant("int"), Constant("str")], Load()), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)

    # Test Optional conversion
    resolver = Resolver("test", {"typing.Optional": "Optional"})
    node = Subscript(Name("Optional", Load()), Constant("int"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)
    assert result.right.value is None

    # Test non-PEP585 name
    resolver = Resolver("test", {"typing.Dict": "dict"})
    node = Subscript(Name("Dict", Load()), Constant("str"), Load())
    result = resolver.visit_Subscript(node)
    assert isinstance(result, Subscript)
    assert isinstance(result.value, Name)
    assert result.value.id == "Dict"


# LLM-generated content at query #15
#--------------------------

```python
def test_Resolver_visit_Constant():
    # Test with non-string constant
    resolver = Resolver("test", {})
    node = Constant(123)
    assert resolver.visit_Constant(node) == node

    # Test with string that is not a valid name
    node = Constant("not_a_valid_name!")
    assert resolver.visit_Constant(node) == node

    # Test with valid name string
    node = Constant("valid_name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Name)
    assert result.id == "valid_name"

    # Test with nested expression
    node = Constant("some.module.name")
    result = resolver.visit_Constant(node)
    assert isinstance(result, Attribute)
    assert result.attr == "name"
    assert isinstance(result.value, Attribute)
    assert result.value.attr == "module"
    assert isinstance(result.value.value, Name)
    assert result.value.value.id == "some"


# LLM-generated content at query #16
#--------------------------

```python
def test_Parser_func_ann():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no self and no class method
    args = [
        arg("x", Name("int", Load())),
        arg("y", Name("str", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "str", "bool"]

    # Test case 2: Function with self and not a class method
    args = [
        arg("self", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "int", "str"]

    # Test case 3: Function with self and is a class method
    args = [
        arg("cls", Name("TestClass", Load())),
        arg("x", Name("int", Load())),
        arg("return", Name("str", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=True))
    assert result == ["type[Self]", "int", "str"]

    # Test case 4: Function with *args and **kwargs
    args = [
        arg("x", Name("int", Load())),
        arg("*", None),
        arg("y", Name("str", Load())),
        arg("**", None),
        arg("kwargs", Name("dict", Load())),
        arg("return", Name("bool", Load()))
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["int", "", "str", "", "dict", "bool"]

    # Test case 5: Function with no annotations
    args = [
        arg("x", None),
        arg("y", None),
        arg("return", None)
    ]
    result = list(parser.func_ann(root, args, has_self=False, cls_method=False))
    assert result == ["Any", "Any", "Any"]

    # Test case 6: Function with mixed annotations
    args = [
        arg("self", Name("TestClass", Load())),
        arg("x", None),
        arg("y", Name("str", Load())),
        arg("return", None)
    ]
    result = list(parser.func_ann(root, args, has_self=True, cls_method=False))
    assert result == ["Self", "Any", "str", "Any"]


# LLM-generated content at query #17
#--------------------------

```python
def test_Parser_imports():
    # Test case 1: Import with no alias
    parser = Parser()
    import_node = Import(names=[alias(name='os')])
    parser.imports('test_module', import_node)
    assert parser.alias['test_module.os'] == 'os'

    # Test case 2: Import with alias
    parser = Parser()
    import_node = Import(names=[alias(name='numpy', asname='np')])
    parser.imports('test_module', import_node)
    assert parser.alias['test_module.np'] == 'numpy'

    # Test case 3: ImportFrom with no level and no alias
    parser = Parser()
    import_from_node = ImportFrom(module='sys', names=[alias(name='path')], level=0)
    parser.imports('test_module', import_from_node)
    assert parser.alias['test_module.path'] == 'sys.path'

    # Test case 4: ImportFrom with level and alias
    parser = Parser()
    import_from_node = ImportFrom(module='os', names=[alias(name='path', asname='osp')], level=1)
    parser.imports('test_module.submodule', import_from_node)
    assert parser.alias['test_module.submodule.osp'] == 'test_module.os.path'


# LLM-generated content at query #18
#--------------------------

```python
def test_walk_body():
    # Test empty body
    assert list(walk_body([])) == []

    # Test simple statements
    stmt1 = Assign(targets=[Name(id='x', ctx=Load())], value=Constant(value=1))
    stmt2 = Expr(value=Call(func=Name(id='print', ctx=Load()), args=[], keywords=[]))
    assert list(walk_body([stmt1, stmt2])) == [stmt1, stmt2]

    # Test If statement
    if_stmt = If(
        test=Constant(value=True),
        body=[Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2))],
        orelse=[Expr(value=Call(func=Name(id='exit', ctx=Load()), args=[], keywords=[]))]
    )
    assert list(walk_body([if_stmt])) == [
        Assign(targets=[Name(id='y', ctx=Load())], value=Constant(value=2)),
        Expr(value=Call(func=Name(id='exit', ctx=Load()), args=[], keywords=[]))
    ]

    # Test Try statement
    try_stmt = Try(
        body=[Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3))],
        handlers=[],
        orelse=[Expr(value=Call(func=Name(id='log', ctx=Load()), args=[], keywords=[]))],
        finalbody=[Expr(value=Call(func=Name(id='cleanup', ctx=Load()), args=[], keywords=[]))]
    )
    assert list(walk_body([try_stmt])) == [
        Assign(targets=[Name(id='z', ctx=Load())], value=Constant(value=3)),
        Expr(value=Call(func=Name(id='log', ctx=Load()), args=[], keywords=[])),
        Expr(value=Call(func=Name(id='cleanup', ctx=Load()), args=[], keywords=[]))
    ]

    # Test nested If and Try
    nested_stmt = If(
        test=Constant(value=True),
        body=[
            Try(
                body=[Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))],
                handlers=[],
                orelse=[],
                finalbody=[]
            )
        ],
        orelse=[]
    )
    assert list(walk_body([nested_stmt])) == [
        Assign(targets=[Name(id='a', ctx=Load())], value=Constant(value=4))
    ]

    # Test mixed statements
    mixed_body = [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5)),
        If(
            test=Constant(value=True),
            body=[Expr(value=Call(func=Name(id='func', ctx=Load()), args=[], keywords=[]))],
            orelse=[]
        ),
        Try(
            body=[],
            handlers=[],
            orelse=[Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))],
            finalbody=[]
        )
    ]
    assert list(walk_body(mixed_body)) == [
        Assign(targets=[Name(id='b', ctx=Load())], value=Constant(value=5)),
        Expr(value=Call(func=Name(id='func', ctx=Load()), args=[], keywords=[])),
        Assign(targets=[Name(id='c', ctx=Load())], value=Constant(value=6))
    ]


# LLM-generated content at query #19
#--------------------------

```python
def test_Parser_func_api():
    parser = Parser()
    root = "test_module"

    # Test case 1: Simple function with no arguments
    func_node = FunctionDef(
        name="simple_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.func_api(root, "test_module.simple_func", func_node.args, None, has_self=False, cls_method=False)
    assert "test_module.simple_func" in parser.doc
    assert "return" in parser.doc["test_module.simple_func"]

    # Test case 2: Function with positional arguments
    func_node = FunctionDef(
        name="pos_args_func",
        args=arguments(
            posonlyargs=[arg(arg="arg1", annotation=Name(id="int", ctx=Load()))],
            args=[arg(arg="arg2", annotation=Name(id="str", ctx=Load()))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.func_api(root, "test_module.pos_args_func", func_node.args, None, has_self=False, cls_method=False)
    assert "test_module.pos_args_func" in parser.doc
    assert "arg1" in parser.doc["test_module.pos_args_func"]
    assert "arg2" in parser.doc["test_module.pos_args_func"]
    assert "return" in parser.doc["test_module.pos_args_func"]

    # Test case 3: Function with keyword arguments
    func_node = FunctionDef(
        name="kw_args_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[arg(arg="kw_arg", annotation=Name(id="float", ctx=Load()))],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.func_api(root, "test_module.kw_args_func", func_node.args, None, has_self=False, cls_method=False)
    assert "test_module.kw_args_func" in parser.doc
    assert "kw_arg" in parser.doc["test_module.kw_args_func"]
    assert "return" in parser.doc["test_module.kw_args_func"]

    # Test case 4: Function with varargs and kwargs
    func_node = FunctionDef(
        name="varargs_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=arg(arg="args", annotation=None),
            kwarg=arg(arg="kwargs", annotation=None)
        ),
        body=[],
        decorator_list=[]
    )
    parser.func_api(root, "test_module.varargs_func", func_node.args, None, has_self=False, cls_method=False)
    assert "test_module.varargs_func" in parser.doc
    assert "*args" in parser.doc["test_module.varargs_func"]
    assert "**kwargs" in parser.doc["test_module.varargs_func"]
    assert "return" in parser.doc["test_module.varargs_func"]

    # Test case 5: Function with return annotation
    func_node = FunctionDef(
        name="return_func",
        args=arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[],
        returns=Name(id="bool", ctx=Load())
    )
    parser.func_api(root, "test_module.return_func", func_node.args, func_node.returns, has_self=False, cls_method=False)
    assert "test_module.return_func" in parser.doc
    assert "return" in parser.doc["test_module.return_func"]
    assert "bool" in parser.doc["test_module.return_func"]

    # Test case 6: Function with self parameter (method)
    func_node = FunctionDef(
        name="method_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="self", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[]
    )
    parser.func_api(root, "test_module.method_func", func_node.args, None, has_self=True, cls_method=False)
    assert "test_module.method_func" in parser.doc
    assert "Self" in parser.doc["test_module.method_func"]
    assert "return" in parser.doc["test_module.method_func"]

    # Test case 7: Function with classmethod decorator
    func_node = FunctionDef(
        name="classmethod_func",
        args=arguments(
            posonlyargs=[],
            args=[arg(arg="cls", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[],
        decorator_list=[Name(id="classmethod", ctx=Load())]
    )
    parser.func_api(root, "test_module.classmethod_func", func_node.args, None, has_self=True, cls_method=True)
    assert "test_module.classmethod_func" in parser.doc
    assert "type[Self]" in parser.doc["test_module.classmethod_func"]
    assert "return" in parser.doc["test_module.classmethod_func"]


# LLM-generated content at query #20
#--------------------------

```python
def test_Parser_api():
    parser = Parser.new(link=True, level=1, toc=False)
    parser.parse("test_module", """
from typing import List

def test_function(arg1: int, arg2: str) -> bool:
    '''Test function docstring.'''
    return True

class TestClass:
    '''Test class docstring.'''

    def method(self, arg: float) -> None:
        '''Test method docstring.'''
        pass

    @staticmethod
    def static_method(arg: List[int]) -> str:
        '''Test static method docstring.'''
        return ""

    @classmethod
    def class_method(cls, arg: dict) -> None:
        '''Test class method docstring.'''
        pass

async def async_function(arg: complex) -> bytes:
    '''Test async function docstring.'''
    return b""
""")

    # Test function API
    assert "test_function" in parser.doc
    assert "def test_function" in parser.doc["test_module.test_function"]
    assert "arg1" in parser.doc["test_module.test_function"]
    assert "arg2" in parser.doc["test_module.test_function"]
    assert "return" in parser.doc["test_module.test_function"]

    # Test class API
    assert "TestClass" in parser.doc
    assert "class TestClass" in parser.doc["test_module.TestClass"]
    assert "method" in parser.doc["test_module.TestClass.method"]
    assert "static_method" in parser.doc["test_module.TestClass.static_method"]
    assert "class_method" in parser.doc["test_module.TestClass.class_method"]

    # Test async function API
    assert "async_function" in parser.doc
    assert "async async_function" in parser.doc["test_module.async_function"]
    assert "arg" in parser.doc["test_module.async_function"]
    assert "return" in parser.doc["test_module.async_function"]

    # Test docstrings
    assert "Test function docstring" in parser.docstring["test_module.test_function"]
    assert "Test class docstring" in parser.docstring["test_module.TestClass"]
    assert "Test method docstring" in parser.docstring["test_module.TestClass.method"]
    assert "Test static method docstring" in parser.docstring["test_module.TestClass.static_method"]
    assert "Test class method docstring" in parser.docstring["test_module.TestClass.class_method"]
    assert "Test async function docstring" in parser.docstring["test_module.async_function"]


